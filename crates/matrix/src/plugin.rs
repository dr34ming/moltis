use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use {
    async_trait::async_trait,
    matrix_sdk::{
        Client,
        config::SyncSettings,
        ruma::OwnedUserId,
    },
    secrecy::ExposeSecret,
    tracing::{info, warn},
};

use moltis_channels::{
    ChannelConfigView, Error as ChannelError, Result as ChannelResult,
    message_log::MessageLog,
    otp::OtpState,
    plugin::{
        ChannelEventSink, ChannelHealthSnapshot, ChannelOutbound, ChannelPlugin, ChannelStatus,
        ChannelStreamOutbound, ChannelThreadContext,
    },
};

use crate::{
    config::MatrixAccountConfig,
    handler,
    outbound::MatrixOutbound,
    state::{AccountState, AccountStateMap},
};

/// Matrix/Element channel plugin for Moltis.
pub struct MatrixPlugin {
    accounts: AccountStateMap,
    outbound: MatrixOutbound,
    message_log: Option<Arc<dyn MessageLog>>,
    event_sink: Option<Arc<dyn ChannelEventSink>>,
}

impl MatrixPlugin {
    pub fn new() -> Self {
        let accounts: AccountStateMap = Arc::new(RwLock::new(HashMap::new()));
        let outbound = MatrixOutbound {
            accounts: Arc::clone(&accounts),
        };
        Self {
            accounts,
            outbound,
            message_log: None,
            event_sink: None,
        }
    }

    pub fn with_message_log(mut self, log: Arc<dyn MessageLog>) -> Self {
        self.message_log = Some(log);
        self
    }

    pub fn with_event_sink(mut self, sink: Arc<dyn ChannelEventSink>) -> Self {
        self.event_sink = Some(sink);
        self
    }
}

impl Default for MatrixPlugin {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ChannelPlugin for MatrixPlugin {
    fn id(&self) -> &str {
        "matrix"
    }

    fn name(&self) -> &str {
        "Matrix"
    }

    async fn start_account(
        &mut self,
        account_id: &str,
        config: serde_json::Value,
    ) -> ChannelResult<()> {
        let cfg: MatrixAccountConfig = serde_json::from_value(config)?;
        if cfg.homeserver.is_empty() {
            return Err(ChannelError::invalid_input("homeserver URL is required"));
        }
        if cfg.access_token.expose_secret().is_empty() {
            return Err(ChannelError::invalid_input("access_token is required"));
        }

        info!(account_id, homeserver = %cfg.homeserver, "starting matrix account");

        // Build Matrix client
        let client = Client::builder()
            .homeserver_url(&cfg.homeserver)
            .build()
            .await
            .map_err(|e| ChannelError::external("matrix client build", e))?;

        // Restore session with access token
        let token = cfg.access_token.expose_secret().clone();
        let device_id_str = cfg.device_id.clone().unwrap_or_else(|| format!("moltis_{account_id}"));

        // Restore session with access token
        let user_id: OwnedUserId = if let Some(uid) = &cfg.user_id {
            uid.parse()
                .map_err(|e: matrix_sdk::ruma::IdParseError| {
                    ChannelError::invalid_input(format!("invalid user_id: {e}"))
                })?
        } else {
            return Err(ChannelError::invalid_input(
                "user_id is required in config (auto-detection not yet supported)",
            ));
        };

        let session = matrix_sdk::authentication::matrix::MatrixSession {
            meta: matrix_sdk::SessionMeta {
                user_id,
                device_id: device_id_str.into(),
            },
            tokens: matrix_sdk::SessionTokens {
                access_token: token,
                refresh_token: None,
            },
        };
        client
            .restore_session(session)
            .await
            .map_err(|e| ChannelError::external("matrix session restore", e))?;

        let bot_user_id = client.user_id().unwrap().to_owned();
        info!(account_id, user_id = %bot_user_id, "matrix session restored");

        let cancel = tokio_util::sync::CancellationToken::new();

        {
            let otp_cooldown = cfg.otp_cooldown_secs;
            let mut accounts = self.accounts.write().unwrap_or_else(|e| e.into_inner());
            accounts.insert(
                account_id.to_string(),
                AccountState {
                    account_id: account_id.to_string(),
                    config: cfg,
                    client: client.clone(),
                    message_log: self.message_log.clone(),
                    event_sink: self.event_sink.clone(),
                    cancel: cancel.clone(),
                    bot_user_id: bot_user_id.to_string(),
                    otp: std::sync::Mutex::new(OtpState::new(otp_cooldown)),
                },
            );
        }

        // Register event handlers
        let accounts_for_msg = Arc::clone(&self.accounts);
        let account_id_msg = account_id.to_string();
        let bot_uid_msg = bot_user_id.clone();
        client.add_event_handler(
            move |ev: matrix_sdk::ruma::events::room::message::OriginalSyncRoomMessageEvent, room: matrix_sdk::Room| {
                let accounts = Arc::clone(&accounts_for_msg);
                let aid = account_id_msg.clone();
                let buid = bot_uid_msg.clone();
                async move {
                    handler::handle_room_message(ev, room, aid, accounts, buid).await;
                }
            },
        );

        let accounts_for_invite = Arc::clone(&self.accounts);
        let account_id_invite = account_id.to_string();
        let bot_uid_invite = bot_user_id.clone();
        client.add_event_handler(
            move |ev: matrix_sdk::ruma::events::room::member::StrippedRoomMemberEvent, room: matrix_sdk::Room| {
                let accounts = Arc::clone(&accounts_for_invite);
                let aid = account_id_invite.clone();
                let buid = bot_uid_invite.clone();
                async move {
                    handler::handle_invite(ev, room, aid, accounts, buid).await;
                }
            },
        );

        // Do initial sync
        info!(account_id, "performing initial sync...");
        client
            .sync_once(SyncSettings::default())
            .await
            .map_err(|e| ChannelError::external("matrix initial sync", e))?;
        info!(account_id, "initial sync complete, starting continuous sync");

        // Spawn continuous sync loop
        let cancel_for_sync = cancel.clone();
        let client_for_sync = client.clone();
        tokio::spawn(async move {
            tokio::select! {
                _ = client_for_sync.sync(SyncSettings::default()) => {
                    warn!("matrix sync loop ended unexpectedly");
                }
                () = cancel_for_sync.cancelled() => {
                    info!("matrix sync loop cancelled");
                }
            }
        });

        Ok(())
    }

    async fn stop_account(&mut self, account_id: &str) -> ChannelResult<()> {
        let cancel = {
            let mut accounts = self.accounts.write().unwrap_or_else(|e| e.into_inner());
            accounts.remove(account_id).map(|s| s.cancel)
        };
        if let Some(cancel) = cancel {
            cancel.cancel();
            info!(account_id, "matrix account stopped");
        } else {
            warn!(account_id, "matrix account not found");
        }
        Ok(())
    }

    fn outbound(&self) -> Option<&dyn ChannelOutbound> {
        Some(&self.outbound)
    }

    fn status(&self) -> Option<&dyn ChannelStatus> {
        Some(self)
    }

    fn has_account(&self, account_id: &str) -> bool {
        let accounts = self.accounts.read().unwrap_or_else(|e| e.into_inner());
        accounts.contains_key(account_id)
    }

    fn account_ids(&self) -> Vec<String> {
        let accounts = self.accounts.read().unwrap_or_else(|e| e.into_inner());
        accounts.keys().cloned().collect()
    }

    fn account_config(&self, account_id: &str) -> Option<Box<dyn ChannelConfigView>> {
        let accounts = self.accounts.read().unwrap_or_else(|e| e.into_inner());
        accounts
            .get(account_id)
            .map(|s| Box::new(s.config.clone()) as Box<dyn ChannelConfigView>)
    }

    fn account_config_json(&self, account_id: &str) -> Option<serde_json::Value> {
        let accounts = self.accounts.read().unwrap_or_else(|e| e.into_inner());
        accounts
            .get(account_id)
            .and_then(|s| serde_json::to_value(&s.config).ok())
    }

    fn update_account_config(
        &self,
        account_id: &str,
        config: serde_json::Value,
    ) -> ChannelResult<()> {
        let parsed: MatrixAccountConfig = serde_json::from_value(config)?;
        let mut accounts = self.accounts.write().unwrap_or_else(|e| e.into_inner());
        if let Some(state) = accounts.get_mut(account_id) {
            state.config = parsed;
            Ok(())
        } else {
            Err(ChannelError::unknown_account(account_id))
        }
    }

    fn shared_outbound(&self) -> Arc<dyn ChannelOutbound> {
        Arc::new(MatrixOutbound {
            accounts: Arc::clone(&self.accounts),
        })
    }

    fn shared_stream_outbound(&self) -> Arc<dyn ChannelStreamOutbound> {
        Arc::new(MatrixOutbound {
            accounts: Arc::clone(&self.accounts),
        })
    }

    fn thread_context(&self) -> Option<&dyn ChannelThreadContext> {
        Some(&self.outbound)
    }
}

#[async_trait]
impl ChannelStatus for MatrixPlugin {
    async fn probe(&self, account_id: &str) -> ChannelResult<ChannelHealthSnapshot> {
        let accounts = self.accounts.read().unwrap_or_else(|e| e.into_inner());
        if let Some(state) = accounts.get(account_id) {
            let connected = state.client.matrix_auth().logged_in();
            let details = if connected {
                format!("syncing as {}", state.bot_user_id)
            } else {
                "not logged in".to_string()
            };
            Ok(ChannelHealthSnapshot {
                connected,
                account_id: state.account_id.clone(),
                details: Some(details),
            })
        } else {
            Ok(ChannelHealthSnapshot {
                connected: false,
                account_id: account_id.to_string(),
                details: Some("account not started".into()),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moltis_channels::{ChannelType, InboundMode};

    #[test]
    fn descriptor_coherence() {
        let plugin = MatrixPlugin::new();
        let desc = ChannelType::Matrix.descriptor();

        assert_eq!(desc.channel_type, ChannelType::Matrix);
        assert_eq!(desc.display_name, "Matrix");
        assert_eq!(desc.capabilities.inbound_mode, InboundMode::GatewayLoop);
        assert!(desc.capabilities.supports_streaming);
        assert!(desc.capabilities.supports_threads);
        assert!(desc.capabilities.supports_reactions);
        assert!(desc.capabilities.supports_otp);
        assert!(!desc.capabilities.supports_interactive); // text fallback
        assert!(!desc.capabilities.supports_voice_ingest);
        assert!(!desc.capabilities.supports_pairing);

        assert!(plugin.thread_context().is_some());
    }

    #[test]
    fn plugin_id_and_name() {
        let plugin = MatrixPlugin::new();
        assert_eq!(plugin.id(), "matrix");
        assert_eq!(plugin.name(), "Matrix");
    }

    #[test]
    fn no_accounts_initially() {
        let plugin = MatrixPlugin::new();
        assert!(!plugin.has_account("test"));
        assert!(plugin.account_ids().is_empty());
    }
}
