use std::sync::Arc;

use {
    matrix_sdk::{
        Room,
        media::{MediaFormat, MediaRequestParameters},
        ruma::{
            OwnedUserId,
            events::room::{
                member::StrippedRoomMemberEvent,
                message::{MessageType, OriginalSyncRoomMessageEvent},
                MediaSource,
            },
        },
    },
    tracing::{debug, info, warn},
};

use moltis_channels::{
    ChannelEvent, ChannelType,
    gating::DmPolicy,
    message_log::MessageLogEntry,
    otp::{OtpInitResult, OtpVerifyResult},
    plugin::{ChannelAttachment, ChannelEventSink, ChannelMessageKind, ChannelMessageMeta, ChannelReplyTarget},
};

use crate::{
    access::{self, is_dm_room},
    state::AccountStateMap,
};

pub async fn handle_room_message(
    ev: OriginalSyncRoomMessageEvent,
    room: Room,
    account_id: String,
    accounts: AccountStateMap,
    bot_user_id: OwnedUserId,
) {
    if ev.sender == bot_user_id {
        return;
    }

    let room_id = room.room_id().to_string();
    let sender_id = ev.sender.to_string();
    let event_id = ev.event_id.to_string();

    let (body, kind, attachments) = match &ev.content.msgtype {
        MessageType::Text(text) => (text.body.clone(), ChannelMessageKind::Text, Vec::new()),
        MessageType::Notice(notice) => (notice.body.clone(), ChannelMessageKind::Text, Vec::new()),
        MessageType::Image(img) => {
            let caption = img.body.clone();
            let media_type = img.info.as_ref()
                .and_then(|i| i.mimetype.clone())
                .unwrap_or_else(|| "image/jpeg".to_string());
            let att = match download_matrix_media(&room, &img.source, &media_type).await {
                Ok(a) => vec![a],
                Err(e) => {
                    warn!("failed to download image from Matrix: {e}");
                    Vec::new()
                }
            };
            (caption, ChannelMessageKind::Photo, att)
        }
        MessageType::Audio(_) => (String::new(), ChannelMessageKind::Audio, Vec::new()),
        MessageType::Video(_) => (String::new(), ChannelMessageKind::Video, Vec::new()),
        MessageType::File(_) => (String::new(), ChannelMessageKind::Document, Vec::new()),
        MessageType::Location(_) => (String::new(), ChannelMessageKind::Location, Vec::new()),
        _ => return,
    };

    if body.is_empty() && attachments.is_empty() && matches!(kind, ChannelMessageKind::Text) {
        return;
    }

    // Snapshot config+state without holding lock across .await
    let (config, message_log, event_sink) = {
        let guard = accounts.read().unwrap_or_else(|e| e.into_inner());
        match guard.get(&account_id) {
            Some(s) => (s.config.clone(), s.message_log.clone(), s.event_sink.clone()),
            None => {
                warn!(account_id, "account state not found");
                return;
            }
        }
    };

    let member_count = room.joined_members_count();
    let chat_type = is_dm_room(member_count);

    if let Err(reason) = access::check_access(&config, &chat_type, &sender_id, &room_id) {
        if matches!(chat_type, moltis_common::types::ChatType::Dm)
            && matches!(reason, access::AccessDenied::NotOnAllowlist)
            && config.otp_self_approval
            && config.dm_policy == DmPolicy::Allowlist
        {
            handle_otp(&body, &sender_id, &account_id, &accounts, &event_sink, &room).await;
            return;
        }
        debug!(account_id, sender = %sender_id, %reason, "access denied");
        return;
    }

    let sender_name = room
        .get_member_no_sync(&ev.sender)
        .await
        .ok()
        .flatten()
        .and_then(|m| m.display_name().map(|s| s.to_string()));

    if let Some(emoji) = &config.ack_reaction {
        let room_clone = room.clone();
        let event_id_clone = ev.event_id.clone();
        let emoji_clone = emoji.clone();
        tokio::spawn(async move {
            use matrix_sdk::ruma::events::{reaction::ReactionEventContent, relation::Annotation};
            let annotation = Annotation::new(event_id_clone, emoji_clone);
            let content = ReactionEventContent::new(annotation);
            if let Err(e) = room_clone.send(content).await {
                warn!("failed to send ack reaction: {e}");
            }
        });
    }

    if let Some(log) = &message_log {
        let _ = log
            .log(MessageLogEntry {
                id: 0,
                account_id: account_id.clone(),
                channel_type: "matrix".into(),
                peer_id: sender_id.clone(),
                username: Some(sender_id.clone()),
                sender_name: sender_name.clone(),
                chat_id: room_id.clone(),
                chat_type: if matches!(chat_type, moltis_common::types::ChatType::Dm) { "dm" } else { "group" }.into(),
                body: body.clone(),
                access_granted: true,
                created_at: unix_now(),
            })
            .await;
    }

    let reply_to = ChannelReplyTarget {
        channel_type: ChannelType::Matrix,
        account_id: account_id.clone(),
        chat_id: room_id.clone(),
        message_id: if config.reply_to_message { Some(event_id.clone()) } else { None },
    };

    let meta = ChannelMessageMeta {
        channel_type: ChannelType::Matrix,
        sender_name: sender_name.clone(),
        username: Some(sender_id.clone()),
        message_kind: Some(kind),
        model: config.model.clone(),
        audio_filename: None,
    };

    if let Some(sink) = &event_sink {
        sink.emit(ChannelEvent::InboundMessage {
            channel_type: ChannelType::Matrix,
            account_id: account_id.clone(),
            peer_id: sender_id.clone(),
            username: Some(sender_id.clone()),
            sender_name: sender_name.clone(),
            message_count: Some(1),
            access_granted: true,
        }).await;

        if attachments.is_empty() {
            sink.dispatch_to_chat(&body, reply_to, meta).await;
        } else {
            sink.dispatch_to_chat_with_attachments(&body, attachments, reply_to, meta).await;
        }
    }
}

async fn handle_otp(
    body: &str,
    sender_id: &str,
    account_id: &str,
    accounts: &AccountStateMap,
    event_sink: &Option<Arc<dyn ChannelEventSink>>,
    room: &Room,
) {
    let trimmed = body.trim();

    if trimmed.len() == 6 && trimmed.chars().all(|c| c.is_ascii_digit()) {
        let result = {
            let guard = accounts.read().unwrap_or_else(|e| e.into_inner());
            if let Some(state) = guard.get(account_id) {
                let mut otp = state.otp.lock().unwrap_or_else(|e| e.into_inner());
                otp.verify(sender_id, trimmed)
            } else {
                return;
            }
        };

        match result {
            OtpVerifyResult::Approved => {
                let _ = send_text(room, "Access granted.").await;
                if let Some(sink) = &event_sink {
                    sink.emit(ChannelEvent::OtpResolved {
                        channel_type: ChannelType::Matrix,
                        account_id: account_id.into(),
                        peer_id: sender_id.into(),
                        username: Some(sender_id.into()),
                        resolution: "approved".into(),
                    }).await;
                }
            }
            OtpVerifyResult::WrongCode { attempts_left } => {
                let msg = format!("Invalid code. {attempts_left} attempts remaining.");
                let _ = send_text(room, &msg).await;
            }
            OtpVerifyResult::Expired => {
                let _ = send_text(room, "Code expired. Send any message for a new one.").await;
            }
            OtpVerifyResult::LockedOut => {
                let _ = send_text(room, "Too many attempts. Please wait.").await;
            }
            OtpVerifyResult::NoPending => {
                // Fall through to initiate
            }
        }
        if !matches!(result, OtpVerifyResult::NoPending) {
            return;
        }
    }

    let result = {
        let guard = accounts.read().unwrap_or_else(|e| e.into_inner());
        if let Some(state) = guard.get(account_id) {
            let mut otp = state.otp.lock().unwrap_or_else(|e| e.into_inner());
            otp.initiate(sender_id, Some(sender_id.into()), None)
        } else {
            return;
        }
    };

    match result {
        OtpInitResult::Created(code) => {
            let expires_at = unix_now() + 300;
            let msg = format!(
                "You're not on the allowlist. A verification code has been generated.\n\
                 Ask the admin to approve code: **{code}**\n\
                 Or enter it here if you have it."
            );
            let _ = send_text(room, &msg).await;
            if let Some(sink) = &event_sink {
                sink.emit(ChannelEvent::OtpChallenge {
                    channel_type: ChannelType::Matrix,
                    account_id: account_id.into(),
                    peer_id: sender_id.into(),
                    username: Some(sender_id.into()),
                    sender_name: Some(sender_id.into()),
                    code,
                    expires_at,
                }).await;
            }
        }
        OtpInitResult::AlreadyPending => {
            let _ = send_text(room, "A verification code is already pending.").await;
        }
        OtpInitResult::LockedOut => {
            let _ = send_text(room, "Too many failed attempts. Please wait.").await;
        }
    }
}

pub async fn handle_invite(
    ev: StrippedRoomMemberEvent,
    room: Room,
    account_id: String,
    accounts: AccountStateMap,
    bot_user_id: OwnedUserId,
) {
    if ev.state_key != bot_user_id {
        return;
    }

    let auto_join = {
        let guard = accounts.read().unwrap_or_else(|e| e.into_inner());
        match guard.get(&account_id) {
            Some(s) => s.config.auto_join,
            None => return,
        }
    };

    if !auto_join {
        debug!(account_id, room = %room.room_id(), "ignoring invite (auto_join=false)");
        return;
    }

    info!(account_id, room = %room.room_id(), inviter = %ev.sender, "auto-joining room");
    if let Err(e) = room.join().await {
        warn!(account_id, room = %room.room_id(), "failed to auto-join: {e}");
    }
}

pub async fn send_text(room: &Room, text: &str) -> Result<(), matrix_sdk::Error> {
    use matrix_sdk::ruma::events::room::message::RoomMessageEventContent;
    let content = RoomMessageEventContent::text_plain(text);
    room.send(content).await?;
    Ok(())
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

async fn download_matrix_media(
    room: &Room,
    source: &MediaSource,
    media_type: &str,
) -> Result<ChannelAttachment, matrix_sdk::Error> {
    let request = MediaRequestParameters {
        source: source.clone(),
        format: MediaFormat::File,
    };
    let data = room.client().media().get_media_content(&request, false).await?;
    info!(size = data.len(), media_type, "downloaded image from Matrix");
    Ok(ChannelAttachment {
        media_type: media_type.to_string(),
        data,
    })
}
