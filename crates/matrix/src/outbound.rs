use std::time::Duration;

use {
    async_trait::async_trait,
    base64::{Engine as _, engine::general_purpose::STANDARD as BASE64},
    matrix_sdk::ruma::{
        self,
        events::{
            reaction::ReactionEventContent,
            relation::Annotation,
            room::message::RoomMessageEventContent,
        },
        OwnedEventId, OwnedRoomId,
    },
    tracing::{debug, warn},
};

use {
    moltis_channels::{
        Error as ChannelError, Result as ChannelResult,
        plugin::{
            ChannelOutbound, ChannelStreamOutbound, ChannelThreadContext,
            InteractiveMessage, StreamEvent, StreamReceiver, ThreadMessage,
        },
    },
    moltis_common::types::ReplyPayload,
};

use crate::state::AccountStateMap;

/// Minimum chars before the first message is sent during streaming.
const STREAM_MIN_INITIAL_CHARS: usize = 30;

/// Throttle interval between edit-in-place updates during streaming.
const STREAM_EDIT_THROTTLE: Duration = Duration::from_millis(500);

/// Typing indicator refresh interval.
const TYPING_REFRESH_INTERVAL: Duration = Duration::from_secs(4);

pub struct MatrixOutbound {
    pub accounts: AccountStateMap,
}

impl MatrixOutbound {
    fn get_room(
        &self,
        account_id: &str,
        room_id_str: &str,
    ) -> ChannelResult<matrix_sdk::Room> {
        let accounts = self.accounts.read().unwrap_or_else(|e| e.into_inner());
        let state = accounts
            .get(account_id)
            .ok_or_else(|| ChannelError::unknown_account(account_id))?;
        let client = state.client.clone();
        drop(accounts);

        let room_id: OwnedRoomId = room_id_str
            .parse()
            .map_err(|e: ruma::IdParseError| ChannelError::invalid_input(e.to_string()))?;

        client
            .get_room(&room_id)
            .ok_or_else(|| ChannelError::unavailable(format!("room not found: {room_id_str}")))
    }
}

#[async_trait]
impl ChannelOutbound for MatrixOutbound {
    async fn send_text(
        &self,
        account_id: &str,
        to: &str,
        text: &str,
        _reply_to: Option<&str>,
    ) -> ChannelResult<()> {
        let room = self.get_room(account_id, to)?;
        let content = RoomMessageEventContent::text_markdown(text);
        room.send(content)
            .await
            .map_err(|e| ChannelError::external("matrix send_text", e))?;
        Ok(())
    }

    async fn send_media(
        &self,
        account_id: &str,
        to: &str,
        payload: &ReplyPayload,
        _reply_to: Option<&str>,
    ) -> ChannelResult<()> {
        let room = self.get_room(account_id, to)?;

        if !payload.text.is_empty() {
            let content = RoomMessageEventContent::text_markdown(&payload.text);
            room.send(content)
                .await
                .map_err(|e| ChannelError::external("matrix send_media text", e))?;
        }

        if let Some(media) = &payload.media {
            if media.url.starts_with("data:") {
                if let Some(comma_pos) = media.url.find(',') {
                    let header = &media.url[5..comma_pos];
                    let mime_str = header.split(';').next().unwrap_or("application/octet-stream");
                    if let Ok(bytes) = BASE64.decode(&media.url[comma_pos + 1..]) {
                        let content_type: mime::Mime = mime_str.parse().unwrap_or(mime::APPLICATION_OCTET_STREAM);
                        let filename = format!("image.{}", mime_to_extension(mime_str));
                        let mut config = matrix_sdk::attachment::AttachmentConfig::new();
                        if let Some(reply_id) = _reply_to {
                            if let Ok(eid) = reply_id.parse::<OwnedEventId>() {
                                config.reply = Some(matrix_sdk::room::reply::Reply {
                                    event_id: eid,
                                    enforce_thread: matrix_sdk::room::reply::EnforceThread::MaybeThreaded,
                                });
                            }
                        }
                        room.send_attachment(&filename, &content_type, bytes, config)
                            .await
                            .map_err(|e| ChannelError::external("matrix send_media image", e))?;
                    }
                }
            } else {
                warn!(account_id, url = %media.url, "non-data-URI media not supported for Matrix upload");
            }
        }

        Ok(())
    }

    async fn send_typing(&self, account_id: &str, to: &str) -> ChannelResult<()> {
        let room = self.get_room(account_id, to)?;
        let _ = room.typing_notice(true).await;
        Ok(())
    }

    async fn send_html(
        &self,
        account_id: &str,
        to: &str,
        html: &str,
        _reply_to: Option<&str>,
    ) -> ChannelResult<()> {
        let room = self.get_room(account_id, to)?;
        let content = RoomMessageEventContent::text_html(html_to_plain(html), html);
        room.send(content)
            .await
            .map_err(|e| ChannelError::external("matrix send_html", e))?;
        Ok(())
    }

    async fn send_interactive(
        &self,
        account_id: &str,
        to: &str,
        message: &InteractiveMessage,
        reply_to: Option<&str>,
    ) -> ChannelResult<()> {
        // Matrix doesn't have native buttons — text fallback
        let mut text = message.text.clone();
        for row in &message.button_rows {
            text.push('\n');
            for btn in row {
                text.push_str(&format!("\n  [{}]", btn.label));
            }
        }
        self.send_text(account_id, to, &text, reply_to).await
    }

    async fn add_reaction(
        &self,
        account_id: &str,
        channel_id: &str,
        message_id: &str,
        emoji: &str,
    ) -> ChannelResult<()> {
        let room = self.get_room(account_id, channel_id)?;
        let event_id: OwnedEventId = message_id
            .parse()
            .map_err(|e: ruma::IdParseError| ChannelError::invalid_input(e.to_string()))?;

        let annotation = Annotation::new(event_id, emoji.to_string());
        let content = ReactionEventContent::new(annotation);
        room.send(content)
            .await
            .map_err(|e| ChannelError::external("matrix add_reaction", e))?;
        Ok(())
    }

    async fn remove_reaction(
        &self,
        account_id: &str,
        _channel_id: &str,
        message_id: &str,
        emoji: &str,
    ) -> ChannelResult<()> {
        debug!(account_id, message_id, emoji, "remove_reaction not yet implemented for Matrix");
        Ok(())
    }
}

#[async_trait]
impl ChannelStreamOutbound for MatrixOutbound {
    async fn send_stream(
        &self,
        account_id: &str,
        to: &str,
        _reply_to: Option<&str>,
        mut stream: StreamReceiver,
    ) -> ChannelResult<()> {
        let room = self.get_room(account_id, to)?;

        let mut buffer = String::new();
        let mut sent_event_id: Option<OwnedEventId> = None;
        let mut last_edit = tokio::time::Instant::now();

        let _ = room.typing_notice(true).await;
        let mut typing_refresh = tokio::time::interval(TYPING_REFRESH_INTERVAL);
        typing_refresh.tick().await;

        loop {
            tokio::select! {
                _ = typing_refresh.tick() => {
                    if sent_event_id.is_none() {
                        let _ = room.typing_notice(true).await;
                    }
                }
                event = stream.recv() => {
                    match event {
                        Some(StreamEvent::Delta(chunk)) => {
                            buffer.push_str(&chunk);

                            if sent_event_id.is_none() && buffer.len() >= STREAM_MIN_INITIAL_CHARS {
                                let content = RoomMessageEventContent::text_markdown(&buffer);
                                match room.send(content).await {
                                    Ok(response) => {
                                        sent_event_id = Some(response.event_id);
                                        last_edit = tokio::time::Instant::now();
                                        let _ = room.typing_notice(false).await;
                                    }
                                    Err(e) => {
                                        warn!("stream initial send failed: {e}");
                                        return Err(ChannelError::external("matrix stream", e));
                                    }
                                }
                            } else if sent_event_id.is_some()
                                && last_edit.elapsed() >= STREAM_EDIT_THROTTLE
                            {
                                if let Some(eid) = &sent_event_id {
                                    let edit = make_edit_content(eid, &buffer);
                                    if let Err(e) = room.send(edit).await {
                                        warn!("stream edit failed: {e}");
                                    }
                                    last_edit = tokio::time::Instant::now();
                                }
                            }
                        }
                        Some(StreamEvent::Done) => {
                            if let Some(eid) = &sent_event_id {
                                let edit = make_edit_content(eid, &buffer);
                                let _ = room.send(edit).await;
                            } else if !buffer.is_empty() {
                                let content = RoomMessageEventContent::text_markdown(&buffer);
                                let _ = room.send(content).await;
                            }
                            let _ = room.typing_notice(false).await;
                            break;
                        }
                        Some(StreamEvent::Error(e)) => {
                            warn!("stream error: {e}");
                            if !buffer.is_empty() {
                                buffer.push_str("\n\n[stream error]");
                                if let Some(eid) = &sent_event_id {
                                    let edit = make_edit_content(eid, &buffer);
                                    let _ = room.send(edit).await;
                                } else {
                                    let content = RoomMessageEventContent::text_markdown(&buffer);
                                    let _ = room.send(content).await;
                                }
                            }
                            let _ = room.typing_notice(false).await;
                            break;
                        }
                        None => {
                            let _ = room.typing_notice(false).await;
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    async fn is_stream_enabled(&self, _account_id: &str) -> bool {
        true
    }
}

#[async_trait]
impl ChannelThreadContext for MatrixOutbound {
    async fn fetch_thread_messages(
        &self,
        account_id: &str,
        channel_id: &str,
        thread_id: &str,
        limit: usize,
    ) -> ChannelResult<Vec<ThreadMessage>> {
        debug!(account_id, channel_id, thread_id, limit, "fetch_thread_messages not yet implemented");
        Ok(Vec::new())
    }
}

/// Create an m.replace edit event content.
fn make_edit_content(
    original_event_id: &OwnedEventId,
    new_body: &str,
) -> RoomMessageEventContent {
    use matrix_sdk::ruma::events::room::message::ReplacementMetadata;
    let new_content = RoomMessageEventContent::text_markdown(new_body);
    let metadata = ReplacementMetadata::new(original_event_id.clone(), None);
    new_content.make_replacement(metadata)
}

/// Simple HTML to plain text conversion.
fn html_to_plain(html: &str) -> String {
    html.replace("<br>", "\n")
        .replace("<br/>", "\n")
        .replace("<br />", "\n")
        .replace("<p>", "")
        .replace("</p>", "\n")
        .replace("<b>", "**")
        .replace("</b>", "**")
        .replace("<strong>", "**")
        .replace("</strong>", "**")
        .replace("<i>", "_")
        .replace("</i>", "_")
        .replace("<em>", "_")
        .replace("</em>", "_")
        .replace("<code>", "`")
        .replace("</code>", "`")
}

fn mime_to_extension(mime: &str) -> &'static str {
    match mime {
        "image/png" => "png",
        "image/gif" => "gif",
        "image/webp" => "webp",
        _ => "jpg",
    }
}
