#!/usr/bin/env bash
# One-time setup on the Oracle Cloud VM (decision U23, tracker U24).
#
# Creates a user that can do exactly one thing: carry an SSH tunnel from the
# GitHub Actions runner to Kite's API (and to api.ipify.org, which reports the
# egress IP).  No shell, no terminal, no other destinations.  The runner's
# Kite calls then leave from this VM's reserved public IP, the one registered
# with Zerodha.  Nothing listens on the internet except SSH (key-only).
#
#   sudo bash setup_tunnel_user.sh "ssh-ed25519 AAAA... centurion-kite-tunnel"
#
# It prints the three values to store in GitHub (see README.md).
set -euo pipefail

PUBKEY="${1:?usage: sudo bash setup_tunnel_user.sh '<the tunnel public key, one line>'}"
USER_NAME="kitetunnel"
DESTS='permitopen="api.kite.trade:443",permitopen="api.ipify.org:443"'

case "$PUBKEY" in
  ssh-ed25519\ *|ssh-rsa\ *|ecdsa-sha2-*) ;;
  *) echo "that does not look like an SSH public key (one line starting ssh-ed25519)" >&2; exit 1 ;;
esac
[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }

NOLOGIN="$(command -v nologin || echo /usr/sbin/nologin)"
id -u "$USER_NAME" >/dev/null 2>&1 || useradd --create-home --shell "$NOLOGIN" "$USER_NAME"

HOME_DIR="$(getent passwd "$USER_NAME" | cut -d: -f6)"
install -d -m 700 -o "$USER_NAME" -g "$USER_NAME" "$HOME_DIR/.ssh"
# restrict: no pty, no agent/X11 forwarding, no ~/.ssh/rc; port-forwarding back
# on, limited by permitopen to the two destinations above.
printf 'restrict,port-forwarding,%s %s\n' "$DESTS" "$PUBKEY" > "$HOME_DIR/.ssh/authorized_keys"
chown "$USER_NAME:$USER_NAME" "$HOME_DIR/.ssh/authorized_keys"
chmod 600 "$HOME_DIR/.ssh/authorized_keys"

cat > /etc/ssh/sshd_config.d/60-kitetunnel.conf <<EOF
# Centurion Kite tunnel (deployment/oracle-proxy): forwarding only, key only.
Match User $USER_NAME
    PasswordAuthentication no
    KbdInteractiveAuthentication no
    AllowTcpForwarding local
    PermitTTY no
    X11Forwarding no
    AllowAgentForwarding no
    GatewayPorts no
EOF
grep -qE '^\s*Include\s+/etc/ssh/sshd_config\.d/\*\.conf' /etc/ssh/sshd_config || \
  echo "WARNING: /etc/ssh/sshd_config has no 'Include /etc/ssh/sshd_config.d/*.conf'; add it" >&2
sshd -t
systemctl reload ssh 2>/dev/null || systemctl reload sshd

IP="$(curl -fsS https://api.ipify.org)"
HOSTKEY="$(cut -d' ' -f1-2 /etc/ssh/ssh_host_ed25519_key.pub)"
cat <<EOF

Done.  Put these in GitHub (Settings -> Secrets and variables -> Actions):
  secret   CENTURION_PROXY_HOST      = $USER_NAME@$IP
  secret   CENTURION_PROXY_HOST_KEY  = $IP $HOSTKEY
  variable CENTURION_KITE_STATIC_IP  = $IP
and register $IP in developers.kite.trade -> Profile -> IP Whitelist.
EOF
