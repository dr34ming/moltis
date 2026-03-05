use std::{
    collections::HashMap,
    sync::{Arc, Mutex, RwLock},
};

use {
    moltis_channels::{ChannelEventSink, message_log::MessageLog, otp::OtpState},
    tokio_util::sync::CancellationToken,
};

use crate::config::MatrixAccountConfig;

/// Shared account state map.
pub type AccountStateMap = Arc<RwLock<HashMap<String, AccountState>>>;

/// Per-account runtime state.
pub struct AccountState {
    pub account_id: String,
    pub config: MatrixAccountConfig,
    pub client: matrix_sdk::Client,
    pub message_log: Option<Arc<dyn MessageLog>>,
    pub event_sink: Option<Arc<dyn ChannelEventSink>>,
    pub cancel: CancellationToken,
    pub bot_user_id: String,
    /// In-memory OTP challenges (std::sync::Mutex — never held across .await).
    pub otp: Mutex<OtpState>,
}
