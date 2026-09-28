# Static IP for Kite orders: Oracle Cloud Always Free (tracker U24)

Since April 2026 Zerodha accepts API orders only from a static IP registered
to your developer account, and each IP can belong to one account only. The
GitHub Actions runner that runs the live session has a different IP every
time, so its Kite calls go through an SSH tunnel to a small Oracle Cloud VM
whose reserved public IP is the registered one. Cost: $0 within Oracle's
Always Free limits.

```
GitHub runner --SSH (key only)--> Oracle VM (reserved IP) --HTTPS--> api.kite.trade
```

Why a tunnel rather than a web proxy: nothing listens on the VM except SSH,
the traffic is encrypted end to end, and the key can only reach
`api.kite.trade:443` (and `api.ipify.org:443`, for the IP check). A stolen
key cannot open a shell or reach anything else. The login page flow does not
use the tunnel: only the evening session's calls do.

## 1. Create the VM (once)

1. Sign up at cloud.oracle.com, home region **India West (Mumbai)**.
2. Convert the account to **Pay As You Go** (Billing → Upgrade). Always Free
   resources stay free; without it Oracle may reclaim an instance that stays
   idle for 7 days (under 20% CPU, network and memory), and a tunnel is idle
   almost all day.
3. Compute → Instances → Create: image **Ubuntu 24.04**, shape
   **VM.Standard.E2.1.Micro** (Always Free; A1.Flex also works), a public
   subnet, and your own SSH key for administration.
4. Networking → Reserved public IPs → Reserve, then on the instance's VNIC
   replace the ephemeral public IP with the reserved one. A reserved IP
   survives the instance: if the VM is ever rebuilt, reattach it and nothing
   at Zerodha changes.

## 2. Make the tunnel key and user (once)

On your own machine, never in chat:

```bash
ssh-keygen -t ed25519 -N "" -C centurion-kite-tunnel -f kite_tunnel
cat kite_tunnel.pub            # one line, used below
```

Copy `setup_tunnel_user.sh` to the VM and run it with the public key:

```bash
scp deployment/oracle-proxy/setup_tunnel_user.sh ubuntu@<reserved-ip>:
ssh ubuntu@<reserved-ip> "sudo bash setup_tunnel_user.sh '$(cat kite_tunnel.pub)'"
```

It prints the values for step 3.

## 3. GitHub and Zerodha (once)

| Where | Name | Value |
|---|---|---|
| GitHub secret | `CENTURION_PROXY_SSH_KEY` | contents of the private key file `kite_tunnel` |
| GitHub secret | `CENTURION_PROXY_HOST` | `kitetunnel@<reserved-ip>` (printed) |
| GitHub secret | `CENTURION_PROXY_HOST_KEY` | `<reserved-ip> ssh-ed25519 AAAA...` (printed) |
| GitHub variable | `CENTURION_KITE_STATIC_IP` | `<reserved-ip>` |
| developers.kite.trade | Profile → IP Whitelist | `<reserved-ip>` (changes allowed once a week) |

Then delete the private key file from your machine or keep it in a password
manager; GitHub holds the working copy.

## 4. What the live step does

When `CENTURION_PROXY_HOST` is set, the "Live book - session" step writes the
key and the pinned host key, opens `ssh -D 127.0.0.1:1080` to the VM and runs
the session with `CENTURION_KITE_PROXY=socks5h://127.0.0.1:1080` (Kite calls
resolve and leave at the VM). Before any order the session reads its egress
IP through the tunnel: if it differs from `CENTURION_KITE_STATIC_IP`, real
orders are refused (a dry run just warns). A host key that does not match
the pinned one stops the connection.

## Checking it by hand

```bash
ssh -i kite_tunnel -N -D 127.0.0.1:1080 kitetunnel@<reserved-ip> &
curl --socks5-hostname 127.0.0.1:1080 https://api.ipify.org   # prints <reserved-ip>
curl --socks5-hostname 127.0.0.1:1080 https://example.com     # refused: not permitted
kill %1
```
