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

    // ── Extract body, kind, attachments, and optional voice audio ────────
    let (body, kind, attachments, voice_audio): (
        String,
        ChannelMessageKind,
        Vec<ChannelAttachment>,
        Option<(Vec<u8>, String)>,
    ) = match &ev.content.msgtype {
        MessageType::Text(text) => {
            (text.body.clone(), ChannelMessageKind::Text, Vec::new(), None)
        }
        MessageType::Notice(notice) => {
            (notice.body.clone(), ChannelMessageKind::Text, Vec::new(), None)
        }
        MessageType::Image(img) => {
            let caption = img.body.clone();
            let media_type = img.info.as_ref()
                .and_then(|i| i.mimetype.clone())
                .unwrap_or_else(|| "image/jpeg".to_string());
            let att = match download_matrix_media(&room, &img.source, &media_type).await {
                Ok(raw) => {
                    // Optimize image for LLM (resize large images, compress)
                    let (data, mt) = match moltis_media::image_ops::optimize_for_llm(&raw.data, None) {
                        Ok(opt) => {
                            if opt.was_resized {
                                info!(
                                    original_size = raw.data.len(),
                                    final_size = opt.data.len(),
                                    dims = %format!("{}x{} -> {}x{}", opt.original_width, opt.original_height, opt.final_width, opt.final_height),
                                    "resized image for LLM"
                                );
                            }
                            (opt.data, opt.media_type)
                        }
                        Err(e) => {
                            warn!(error = %e, "failed to optimize image, using original");
                            (raw.data, raw.media_type)
                        }
                    };
                    vec![ChannelAttachment { media_type: mt, data }]
                }
                Err(e) => {
                    warn!("failed to download image from Matrix: {e}");
                    Vec::new()
                }
            };
            (caption, ChannelMessageKind::Photo, att, None)
        }
        MessageType::Audio(audio) => {
            let mime = audio.info.as_ref()
                .and_then(|i| i.mimetype.clone())
                .unwrap_or_else(|| "audio/ogg".to_string());
            let is_voice = audio.voice.is_some();
            match download_matrix_media(&room, &audio.source, &mime).await {
                Ok(att) => {
                    if is_voice {
                        let format = mime_to_audio_format(&mime);
                        if let Some(ref sink) = event_sink {
                            if !sink.voice_stt_available().await {
                                ("[Voice message - transcription not available]".to_string(), ChannelMessageKind::Audio, Vec::new(), Some((att.data, format)))
                            } else {
                                match sink.transcribe_voice(&att.data, &format).await {
                                    Ok(text) if !text.trim().is_empty() => {
                                        debug!(text_len = text.len(), "voice transcription successful");
                                        (text, ChannelMessageKind::Audio, Vec::new(), Some((att.data, format)))
                                    }
                                    Ok(_) => {
                                        warn!("voice transcription returned empty text");
                                        ("[Voice message - could not transcribe]".to_string(), ChannelMessageKind::Audio, Vec::new(), Some((att.data, format)))
                                    }
                                    Err(e) => {
                                        warn!(error = %e, "voice transcription failed");
                                        ("[Voice message - transcription failed]".to_string(), ChannelMessageKind::Audio, Vec::new(), Some((att.data, format)))
                                    }
                                }
                            }
                        } else {
                            ("[Voice message]".to_string(), ChannelMessageKind::Audio, Vec::new(), None)
                        }
                    } else {
                        (audio.body.clone(), ChannelMessageKind::Audio, vec![att], None)
                    }
                }
                Err(e) => {
                    warn!("failed to download audio from Matrix: {e}");
                    ("[Audio - download failed]".to_string(), ChannelMessageKind::Audio, Vec::new(), None)
                }
            }
        }
        MessageType::Video(video) => {
            let desc = format!("[Video: {}]", video.body);
            (desc, ChannelMessageKind::Video, Vec::new(), None)
        }
        MessageType::File(file) => {
            let mime = file.info.as_ref()
                .and_then(|i| i.mimetype.clone())
                .unwrap_or_else(|| "application/octet-stream".to_string());
            match download_matrix_media(&room, &file.source, &mime).await {
                Ok(att) => {
                    if mime.starts_with("text/") || mime == "application/json" || mime == "application/xml" {
                        // Text-based files: embed content in body instead of attachment
                        let text = if att.data.len() > MAX_TEXT_EMBED_BYTES {
                            let s = String::from_utf8_lossy(&att.data);
                            let truncated: String = s.chars().take(MAX_TEXT_EMBED_BYTES).collect();
                            format!("{}\n... [truncated, {} bytes total]", truncated, att.data.len())
                        } else {
                            String::from_utf8_lossy(&att.data).into_owned()
                        };
                        let body = format!("[File: {}]\n```\n{}\n```", file.body, text);
                        (body, ChannelMessageKind::Document, Vec::new(), None)
                    } else if mime.starts_with("image/") {
                        // Image files sent as m.file: treat as image attachment
                        (file.body.clone(), ChannelMessageKind::Photo, vec![att], None)
                    } else {
                        // Binary files: just mention the filename
                        (format!("[File: {} ({})]", file.body, mime), ChannelMessageKind::Document, Vec::new(), None)
                    }
                }
                Err(e) => {
                    warn!("failed to download file from Matrix: {e}");
                    (format!("[File: {} - download failed]", file.body), ChannelMessageKind::Document, Vec::new(), None)
                }
            }
        }
        MessageType::Location(loc) => {
            (loc.body.clone(), ChannelMessageKind::Location, Vec::new(), None)
        }
        _ => return,
    };

    if body.is_empty() && attachments.is_empty() && matches!(kind, ChannelMessageKind::Text) {
        return;
    }

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

    let audio_filename = voice_audio.as_ref().map(|(_, fmt)| format!("voice.{fmt}"));

    let meta = ChannelMessageMeta {
        channel_type: ChannelType::Matrix,
        sender_name: sender_name.clone(),
        username: Some(sender_id.clone()),
        message_kind: Some(kind),
        model: config.model.clone(),
        audio_filename,
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

/// Maximum file size to download from Matrix (10 MB).
const MAX_MEDIA_DOWNLOAD_BYTES: usize = 10 * 1024 * 1024;

/// Maximum text file size to embed in body (32 KB).
const MAX_TEXT_EMBED_BYTES: usize = 32 * 1024;

enum MediaFetchError {
    Sdk(matrix_sdk::Error),
    TooLarge { size: usize },
}

impl std::fmt::Display for MediaFetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sdk(e) => write!(f, "{e}"),
            Self::TooLarge { size } => write!(f, "media too large: {size} bytes (limit {MAX_MEDIA_DOWNLOAD_BYTES})"),
        }
    }
}

impl From<matrix_sdk::Error> for MediaFetchError {
    fn from(e: matrix_sdk::Error) -> Self { Self::Sdk(e) }
}

async fn download_matrix_media(
    room: &Room,
    source: &MediaSource,
    media_type: &str,
) -> Result<ChannelAttachment, MediaFetchError> {
    let request = MediaRequestParameters {
        source: source.clone(),
        format: MediaFormat::File,
    };
    let data = room.client().media().get_media_content(&request, false).await?;
    if data.len() > MAX_MEDIA_DOWNLOAD_BYTES {
        warn!(size = data.len(), media_type, "media too large, skipping");
        return Err(MediaFetchError::TooLarge { size: data.len() });
    }
    info!(size = data.len(), media_type, "downloaded media from Matrix");
    Ok(ChannelAttachment {
        media_type: media_type.to_string(),
        data,
    })
}

/// Map MIME type to simple audio format string for STT.
fn mime_to_audio_format(mime: &str) -> String {
    match mime {
        "audio/ogg" => "ogg",
        "audio/webm" => "webm",
        "audio/mp4" | "audio/m4a" => "m4a",
        "audio/mpeg" | "audio/mp3" => "mp3",
        "audio/wav" | "audio/x-wav" => "wav",
        "audio/flac" => "flac",
        _ => "ogg",
    }.to_string()
}
