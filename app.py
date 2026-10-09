import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.ensemble import IsolationForest

app = Flask(__name__)

# Maximum uploaded file size: 5 MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# CSV file location
LOG_FILE = os.path.join("logs", "uploaded_security_logs.csv")



def analyze_logs():
    """Load security logs and create rule-based and AI alerts."""
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(), []

    try:
        logs = pd.read_csv(LOG_FILE)
    except Exception as e:
        print("Error reading CSV:", e)
        return pd.DataFrame(), []

    required_columns = ["timestamp", "ip", "event", "status", "user"]
    missing_columns = [column for column in required_columns if column not in logs.columns]

    if missing_columns:
        alert = {
            "ip": "N/A",
            "threat": "Invalid Log File",
            "risk_score": 0,
            "risk_level": "LOW",
            "evidence": "Missing columns: " + ", ".join(missing_columns),
            "recommendation": (
                "Upload a CSV containing timestamp, ip, event, status and user columns."
            ),
        }
        return logs, [alert]

    # Normalize values used for detection without changing the original log display.
    events = logs["event"].astype(str).str.strip().str.lower()
    statuses = logs["status"].astype(str).str.strip().str.lower()
    alerts = []

    # ================= CREDENTIAL ABUSE DETECTION =================
    # Detect failed login attempts targeting 3 or more distinct accounts from one IP.
    failed_login_mask = (events == "login") & (statuses == "failed")
    failed_login_events = logs.loc[failed_login_mask].copy()
    failed_login_events["user"] = failed_login_events["user"].fillna("").astype(str)
    users_per_ip = failed_login_events.groupby("ip")["user"].nunique()

    for ip, user_count in users_per_ip.items():
        if user_count >= 3:
            if user_count >= 5:
                risk_score = 85
                risk_level = "CRITICAL"
            else:
                risk_score = 70
                risk_level = "HIGH"

            alerts.append({
                "ip": str(ip),
                "threat": "Possible Credential Abuse Attack",
                "risk_score": risk_score,
                "risk_level": risk_level,
                "evidence": (
                    f"Failed login attempts involved {user_count} distinct user accounts "
                    "from the same IP address."
                ),
                "recommendation": (
                    "Review the targeted accounts, investigate the source IP, "
                    "and consider additional account protections."
                ),
            })

    # ================= BRUTE FORCE DETECTION =================
    failed_count = failed_login_events.groupby("ip").size()

    for ip, count in failed_count.items():
        if count >= 10:
            risk_score = 95
            risk_level = "CRITICAL"
        elif count >= 5:
            risk_score = 75
            risk_level = "HIGH"
        elif count >= 3:
            risk_score = 50
            risk_level = "MEDIUM"
        else:
            continue

        alerts.append({
            "ip": str(ip),
            "threat": "Possible Brute Force Attack",
            "risk_score": risk_score,
            "risk_level": risk_level,
            "evidence": f"{count} failed login attempts",
            "recommendation": (
                "Investigate the source IP and review the affected account."
            ),
        })

    # ================= UNAUTHORIZED FILE ACCESS DETECTION =================
    denied_access_mask = (events == "file_access") & (statuses == "denied")
    denied_access_events = logs.loc[denied_access_mask]
    denied_count = denied_access_events.groupby("ip").size()

    for ip, count in denied_count.items():
        if count >= 5:
            risk_score = 90
            risk_level = "CRITICAL"
        elif count >= 3:
            risk_score = 70
            risk_level = "HIGH"
        else:
            continue

        alerts.append({
            "ip": str(ip),
            "threat": "Possible Unauthorized Access",
            "risk_score": risk_score,
            "risk_level": risk_level,
            "evidence": f"{count} denied file-access attempts",
            "recommendation": (
                "Review the user's permissions and investigate the access attempts."
            ),
        })

    # ================= AI ANOMALY DETECTION =================
    ip_addresses = logs["ip"].dropna().astype(str).unique()
    features = []

    for ip in ip_addresses:
        ip_logs = logs[logs["ip"].astype(str) == ip]
        ip_events = ip_logs["event"].astype(str).str.strip().str.lower()
        ip_statuses = ip_logs["status"].astype(str).str.strip().str.lower()

        failed_login_count = int(((ip_events == "login") & (ip_statuses == "failed")).sum())
        denied_access_count = int(
            ((ip_events == "file_access") & (ip_statuses == "denied")).sum()
        )
        total_activity = len(ip_logs)
        features.append([failed_login_count, denied_access_count, total_activity])

    feature_data = pd.DataFrame(
        features,
        columns=["failed_logins", "denied_access", "total_activity"],
    )

    if len(feature_data) >= 2:
        try:
            model = IsolationForest(
                n_estimators=100,
                contamination="auto",
                random_state=42,
            )
            model.fit(feature_data)
            predictions = model.predict(feature_data)

            for ip, prediction in zip(ip_addresses, predictions):
                if prediction == -1:
                    # Avoid adding a generic AI alert when this IP already has a
                    # more specific rule-based alert.
                    already_has_alert = any(str(alert["ip"]) == str(ip) for alert in alerts)
                    if not already_has_alert:
                        ip_logs = logs[logs["ip"].astype(str) == ip]
                        ip_events = ip_logs["event"].astype(str).str.strip().str.lower()
                        ip_statuses = ip_logs["status"].astype(str).str.strip().str.lower()
                        failed = int(((ip_events == "login") & (ip_statuses == "failed")).sum())
                        denied = int(
                            ((ip_events == "file_access") & (ip_statuses == "denied")).sum()
                        )
                        alerts.append({
                            "ip": str(ip),
                            "threat": "AI Detected Anomalous Activity",
                            "risk_score": 60,
                            "risk_level": "MEDIUM",
                            "evidence": (
                                "Isolation Forest flagged an unusual pattern: "
                                f"{failed} failed logins, {denied} denied accesses, "
                                f"and {len(ip_logs)} total activities."
                            ),
                            "recommendation": (
                                "Investigate this IP's activity and review related logs "
                                "before deciding on a response."
                            ),
                        })
        except Exception as e:
            print("AI anomaly detection error:", e)

    return logs, alerts


# ================= DASHBOARD =================
@app.route("/")
def dashboard():
    logs, alerts = analyze_logs()
    total_logs = len(logs)
    total_alerts = len(alerts)
    high_risk = sum(1 for alert in alerts if alert["risk_level"] == "HIGH")
    critical = sum(1 for alert in alerts if alert["risk_level"] == "CRITICAL")
    medium_risk = sum(1 for alert in alerts if alert["risk_level"] == "MEDIUM")
    low_risk = sum(1 for alert in alerts if alert["risk_level"] == "LOW")

    investigations = []
    if not logs.empty and "ip" in logs.columns:
        for ip in logs["ip"].dropna().astype(str).unique():
            ip_logs = logs[logs["ip"].astype(str) == ip]
            ip_events = ip_logs["event"].astype(str).str.strip().str.lower()
            ip_statuses = ip_logs["status"].astype(str).str.strip().str.lower()
            failed_logins = int(((ip_events == "login") & (ip_statuses == "failed")).sum())
            denied_access = int(
                ((ip_events == "file_access") & (ip_statuses == "denied")).sum()
            )
            if failed_logins >= 3 or denied_access >= 3:
                investigations.append({
                    "ip": ip,
                    "failed_logins": failed_logins,
                    "denied_access": denied_access,
                })

    return render_template(
        "index.html",
        total_logs=total_logs,
        total_alerts=total_alerts,
        high_risk=high_risk,
        critical=critical,
        medium_risk=medium_risk,
        low_risk=low_risk,
        alerts=alerts,
        investigations=investigations,
    )


# ================= CSV UPLOAD =================

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")

    if not file or file.filename == "":
        return "Please select a CSV file.", 400

    if not file.filename.lower().endswith(".csv"):
        return "Invalid file type. Please upload a CSV file.", 400

    try:
        df = pd.read_csv(file)

        if df.empty:
            return "The uploaded CSV file is empty.", 400

        # Check required columns
        missing_columns = REQUIRED_COLUMNS - set(df.columns)

        if missing_columns:
            return (
                "Invalid CSV. Missing columns: "
                + ", ".join(sorted(missing_columns)),
                400
            )

        # Save only after validation succeeds
        df.to_csv(LOG_FILE, index=False)

        return redirect(url_for("dashboard"))

    except (pd.errors.ParserError, UnicodeDecodeError):
        return "Unable to read this CSV file. Please check its format.", 400
    except Exception as e:
        app.logger.exception("CSV upload failed")
    return f"Error: {type(e).__name__}: {e}", 500

# ================= INVESTIGATION PAGE =================
@app.route("/investigate/<path:ip>")
def investigate(ip):
    logs, alerts = analyze_logs()
    if logs.empty:
        return "No log data available. Please upload a valid CSV file.", 404

    logs["ip"] = logs["ip"].astype(str)
    ip = str(ip)
    ip_logs = logs[logs["ip"] == ip]
    if ip_logs.empty:
        return f"No logs found for IP address: {ip}", 404

    ip_events = ip_logs["event"].astype(str).str.strip().str.lower()
    ip_statuses = ip_logs["status"].astype(str).str.strip().str.lower()
    failed_logins = int(((ip_events == "login") & (ip_statuses == "failed")).sum())
    denied_access = int(
        ((ip_events == "file_access") & (ip_statuses == "denied")).sum()
    )
    ip_alerts = [alert for alert in alerts if str(alert["ip"]) == ip]
    related_logs = ip_logs.to_dict(orient="records")

    return render_template(
        "investigate.html",
        ip=ip,
        total_activity=len(ip_logs),
        failed_logins=failed_logins,
        denied_access=denied_access,
        alerts=ip_alerts,
        logs=related_logs,
    )


# ================= START FLASK APPLICATION =================
if __name__ == "__main__":
    print("========================================")
    print("              LOGSHIELD")
    print("       AI Security Log Analyzer")
    print("========================================")
    print("\nStarting Flask server...")
    app.run(host="127.0.0.1", port=5000, debug=True)
