# Sourced by the workflows that call Kite (live session, kill switch): opens the SSH tunnel Kite calls leave
# through, from the primary static-IP host, else from the secondary (Zerodha allows a primary and a secondary
# registered IP, tracker ZR2).  A host counts only when a call through it leaves from its registered IP.
#
#   . deployment/oracle-proxy/static_ip_tunnel.sh && kite_tunnel
#
# Reads CENTURION_PROXY_SSH_KEY, CENTURION_PROXY_HOST / _HOST_KEY / CENTURION_KITE_STATIC_IP and, for the
# backup, CENTURION_PROXY_HOST_2 / CENTURION_PROXY_HOST_KEY_2 / CENTURION_KITE_STATIC_IP_2 (the same tunnel
# key).  On success exports CENTURION_KITE_PROXY and CENTURION_KITE_STATIC_IP for the host in use, so the
# session's own egress check (daily_login.check_egress) still refuses real orders from any other IP.  Returns 1
# when neither host works.  IPs are never printed: the logs are public.

kite_tunnel() {
  local i host hostkey ip name port seen pid
  install -d -m 700 "$HOME/.ssh"
  (umask 077; printf '%s\n' "$CENTURION_PROXY_SSH_KEY" > "$HOME/.ssh/kite_tunnel")
  for i in 1 2; do
    if [ "$i" = 1 ]; then
      host="${CENTURION_PROXY_HOST:-}"; hostkey="${CENTURION_PROXY_HOST_KEY:-}"; ip="${CENTURION_KITE_STATIC_IP:-}"
      name=CENTURION_KITE_STATIC_IP
    else
      host="${CENTURION_PROXY_HOST_2:-}"; hostkey="${CENTURION_PROXY_HOST_KEY_2:-}"; ip="${CENTURION_KITE_STATIC_IP_2:-}"
      name=CENTURION_KITE_STATIC_IP_2
    fi
    [ -n "$host" ] || continue
    if [ -z "$ip" ]; then
      echo "::warning::static IP host $i has no registered IP set ($name): skipped"
      continue
    fi
    port=$((1079 + i))
    printf '%s\n' "$hostkey" > "$HOME/.ssh/known_hosts_kite_$i"
    if ! timeout 30 ssh -i "$HOME/.ssh/kite_tunnel" -o UserKnownHostsFile="$HOME/.ssh/known_hosts_kite_$i" \
        -o StrictHostKeyChecking=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
        -o BatchMode=yes -o ConnectTimeout=20 -f -N -D "127.0.0.1:$port" "$host"; then
      echo "::warning::static IP host $i: the tunnel did not open"
      continue
    fi
    seen="$(curl -fsS --max-time 15 --socks5-hostname "127.0.0.1:$port" https://api.ipify.org || true)"
    if [ "$seen" = "$ip" ]; then
      export CENTURION_KITE_PROXY="socks5h://127.0.0.1:$port" CENTURION_KITE_STATIC_IP="$ip"
      [ "$i" = 2 ] && echo "::warning::the primary static IP is down: Kite calls use the secondary"
      return 0
    fi
    echo "::warning::static IP host $i: calls through it do not leave from its registered IP"
    pid="$(pgrep -f -- "-D 127.0.0.1:$port" || true)"
    [ -n "$pid" ] && kill $pid
  done
  echo "::error::no static-IP tunnel works: Kite calls are not proxied, so real orders are refused"
  return 1
}
