use {
    moltis_channels::{
        config_view::ChannelConfigView,
        gating::{DmPolicy, GroupPolicy},
    },
    secrecy::{ExposeSecret, Secret},
    serde::{Deserialize, Serialize},
};

/// Configuration for a single Matrix account.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MatrixAccountConfig {
    /// Homeserver URL (e.g. "https://matrix.ponderosa.co").
    pub homeserver: String,

    /// Access token for authentication.
    #[serde(serialize_with = "serialize_secret")]
    pub access_token: Secret<String>,

    /// Matrix user ID (auto-detected from whoami if not set).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,

    /// Device ID for session persistence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_id: Option<String>,

    /// DM access policy.
    pub dm_policy: DmPolicy,

    /// Room (group) access policy.
    pub room_policy: GroupPolicy,

    /// Room allowlist (room IDs or aliases).
    pub room_allowlist: Vec<String>,

    /// User allowlist (Matrix user IDs).
    pub user_allowlist: Vec<String>,

    /// Auto-join rooms on invite.
    pub auto_join: bool,

    /// Default model ID for this account.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Provider name associated with `model`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_provider: Option<String>,

    /// Send responses as replies to the original message.
    pub reply_to_message: bool,

    /// Emoji reaction added while processing. None = disabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ack_reaction: Option<String>,

    /// Enable end-to-end encryption support.
    pub e2ee: bool,

    /// OTP self-approval for non-allowlisted DM users.
    pub otp_self_approval: bool,

    /// Cooldown in seconds after 3 failed OTP attempts.
    pub otp_cooldown_secs: u64,
}

impl Default for MatrixAccountConfig {
    fn default() -> Self {
        Self {
            homeserver: String::new(),
            access_token: Secret::new(String::new()),
            user_id: None,
            device_id: None,
            dm_policy: DmPolicy::Allowlist,
            room_policy: GroupPolicy::Allowlist,
            room_allowlist: Vec::new(),
            user_allowlist: Vec::new(),
            auto_join: true,
            model: None,
            model_provider: None,
            reply_to_message: true,
            ack_reaction: Some("\u{1f440}".into()), // 👀
            e2ee: true,
            otp_self_approval: true,
            otp_cooldown_secs: 300,
        }
    }
}

impl std::fmt::Debug for MatrixAccountConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatrixAccountConfig")
            .field("homeserver", &self.homeserver)
            .field("access_token", &"[REDACTED]")
            .field("user_id", &self.user_id)
            .field("device_id", &self.device_id)
            .field("dm_policy", &self.dm_policy)
            .field("room_policy", &self.room_policy)
            .field("room_allowlist", &self.room_allowlist)
            .field("user_allowlist", &self.user_allowlist)
            .field("auto_join", &self.auto_join)
            .field("model", &self.model)
            .field("model_provider", &self.model_provider)
            .field("reply_to_message", &self.reply_to_message)
            .field("ack_reaction", &self.ack_reaction)
            .field("e2ee", &self.e2ee)
            .finish()
    }
}

impl ChannelConfigView for MatrixAccountConfig {
    fn allowlist(&self) -> &[String] {
        &self.user_allowlist
    }

    fn group_allowlist(&self) -> &[String] {
        &self.room_allowlist
    }

    fn dm_policy(&self) -> DmPolicy {
        self.dm_policy.clone()
    }

    fn group_policy(&self) -> GroupPolicy {
        self.room_policy.clone()
    }

    fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn model_provider(&self) -> Option<&str> {
        self.model_provider.as_deref()
    }
}

fn serialize_secret<S: serde::Serializer>(
    secret: &Secret<String>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(secret.expose_secret())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_round_trip() {
        let json = serde_json::json!({
            "homeserver": "https://matrix.example.com",
            "access_token": "syt_test_token",
            "dm_policy": "allowlist",
            "room_policy": "allowlist",
            "room_allowlist": ["!room:example.com"],
            "user_allowlist": ["@alice:example.com"],
            "auto_join": true,
            "reply_to_message": true,
            "ack_reaction": "\u{1f440}",
            "e2ee": true,
        });
        let cfg: MatrixAccountConfig =
            serde_json::from_value(json).expect("parse failed");
        assert_eq!(cfg.homeserver, "https://matrix.example.com");
        assert_eq!(cfg.dm_policy, DmPolicy::Allowlist);
        assert_eq!(cfg.room_allowlist, vec!["!room:example.com"]);

        let value = serde_json::to_value(&cfg).expect("serialize failed");
        let _: MatrixAccountConfig =
            serde_json::from_value(value).expect("re-parse failed");
    }

    #[test]
    fn config_defaults() {
        let cfg = MatrixAccountConfig::default();
        assert_eq!(cfg.dm_policy, DmPolicy::Allowlist);
        assert_eq!(cfg.room_policy, GroupPolicy::Allowlist);
        assert!(cfg.auto_join);
        assert!(cfg.reply_to_message);
        assert_eq!(cfg.ack_reaction.as_deref(), Some("\u{1f440}"));
        assert!(cfg.e2ee);
    }

    #[test]
    fn debug_redacts_token() {
        let cfg = MatrixAccountConfig {
            access_token: Secret::new("super-secret".into()),
            ..Default::default()
        };
        let debug = format!("{cfg:?}");
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("super-secret"));
    }
}
