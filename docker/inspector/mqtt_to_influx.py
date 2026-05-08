import paho.mqtt.client as mqtt
import json
from datetime import datetime, timezone
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# ===== إعدادات InfluxDB =====
import os

INFLUX_URL    = os.getenv("INFLUX_URL",    "http://localhost:8086")
INFLUX_TOKEN  = os.getenv("INFLUX_TOKEN",  "my-super-secret-token")
INFLUX_ORG    = os.getenv("INFLUX_ORG",    "PCB_Factory")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "pcb_inspection")

BROKER = "broker.hivemq.com"
PORT   = 1883
TOPIC  = "pcb/inspection/line1"

# ===== الاتصال بـ InfluxDB =====
influx_client = InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG
)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)
print("✅ متصل بـ InfluxDB")

# ===== حفظ النتيجة في InfluxDB =====
def save_to_influx(data):
    try:
        point = (
            Point("pcb_inspection")
            .tag("line",      "line1")
            .tag("pass_fail", data["pass_fail"])
            .field("total_defects", data["total_defects"])
            .field("inference_ms",  data["inference_ms"])
            .field("board_id",      data["board_id"])
            .time(datetime.now(timezone.utc))
        )

        # حفظ كل عيب بشكل منفصل
        for d in data.get("defects", []):
            defect_point = (
                Point("pcb_defects")
                .tag("line",       "line1")
                .tag("class_name", d["class"])
                .field("confidence", d["conf"])
                .field("board_id",   data["board_id"])
                .time(datetime.now(timezone.utc))
            )
            write_api.write(bucket=INFLUX_BUCKET, record=defect_point)

        write_api.write(bucket=INFLUX_BUCKET, record=point)
        return True

    except Exception as e:
        print(f"❌ خطأ في InfluxDB: {e}")
        return False

# ===== استقبال MQTT وحفظ في InfluxDB =====
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"✅ متصل بـ MQTT broker")
        client.subscribe(TOPIC, qos=1)
        print(f"👂 يستمع على: {TOPIC}")
        print("-" * 55)
    else:
        print(f"❌ فشل MQTT (كود {rc})")

def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())

    # حفظ في InfluxDB
    saved = save_to_influx(data)
    icon  = "💾" if saved else "⚠️"

    # طباعة
    status = "🔴 FAIL" if data["pass_fail"] == "FAIL" else "🟢 PASS"
    print(f"{status}  |  {data['board_id']}  |  "
          f"{data['total_defects']} defects  |  "
          f"{data['inference_ms']}ms  {icon}")

# ===== تشغيل =====
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="influx_bridge")
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, keepalive=60)

print("🚀 Bridge يعمل — MQTT → InfluxDB")
client.loop_forever()