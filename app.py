from flask import Flask, request, jsonify
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import io
import base64
import json
import os

app = Flask(__name__)

# ============================================================
# تحميل النموذج
# ============================================================
MODEL_PATH     = "olive_model.tflite"
CLASS_MAP_PATH = "class_mapping.json"
IMG_SIZE       = 224

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# تحميل class_mapping من ملف JSON إن وُجد
if os.path.exists(CLASS_MAP_PATH):
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
    CLASS_INDEX_TO_KEY = {int(k): v for k, v in raw_map.items()}
    print(f"✅ class_mapping محمّل: {CLASS_INDEX_TO_KEY}")
else:
    CLASS_INDEX_TO_KEY = {
        0: "Healthy",
        1: "aculus_olearius",
        2: "olive_knot",
        3: "olive_peacock_spot",
    }
    print("⚠️  class_mapping.json غير موجود، يستخدم الترتيب الافتراضي")

# ============================================================
# معلومات العرض لكل فئة
# ============================================================
CLASS_DISPLAY = {
    "Healthy": {
        "ar": "ورقة سليمة",
        "en": "Healthy Leaf",
        "is_diseased": False,
    },
    "aculus_olearius": {
        "ar": "أكاروس الزيتون",
        "en": "Aculus Olearius",
        "is_diseased": True,
    },
    "olive_knot": {
        "ar": "عقدة الزيتون",
        "en": "Olive Knot",
        "is_diseased": True,
    },
    "olive_peacock_spot": {
        "ar": "عين الطاووس",
        "en": "Peacock Spot",
        "is_diseased": True,
    },
}

# ============================================================
# التوصيات
# ============================================================
def get_recommendations(class_key: str, confidence: float) -> dict:

    if class_key == "Healthy":
        return {
            "description_ar": "الورقة بصحة ممتازة ولا تحتاج أي تدخل.",
            "description_en": "The leaf is in excellent health and requires no intervention.",
            "severity_ar": "سليم",    "severity_en": "Healthy",
            "urgency_ar":  "لا يوجد", "urgency_en":  "None",
            "recommendations_ar": [
                "استمر في الري المنتظم",
                "تأكد من التسميد الدوري بالأسمدة العضوية",
                "راقب الأشجار بشكل دوري للكشف المبكر",
                "حافظ على التهوية الجيدة بين الفروع",
            ],
            "recommendations_en": [
                "Continue regular irrigation",
                "Ensure periodic fertilisation with organic fertilisers",
                "Monitor trees regularly for early detection",
                "Maintain good ventilation between branches",
            ],
        }

    if class_key == "aculus_olearius":
        if confidence >= 90:
            return {
                "description_ar": "إصابة شديدة بأكاروس الزيتون، يُنصح بالتدخل الفوري.",
                "description_en": "Severe Aculus Olearius infestation. Immediate action is recommended.",
                "severity_ar": "شديد",      "severity_en": "Severe",
                "urgency_ar":  "عاجل جداً", "urgency_en":  "Urgent",
                "recommendations_ar": [
                    "رش مبيد أكاروسي فوراً (Abamectin أو Spiromesifen)",
                    "عزل الشجرة المصابة عن باقي الأشجار",
                    "قطع وإتلاف الفروع الشديدة الإصابة",
                    "كرر الرش بعد 10 أيام",
                ],
                "recommendations_en": [
                    "Spray acaricide immediately (Abamectin or Spiromesifen)",
                    "Isolate the infected tree from others",
                    "Prune and destroy severely infested branches",
                    "Repeat spraying after 10 days",
                ],
            }
        elif confidence >= 70:
            return {
                "description_ar": "إصابة متوسطة بأكاروس الزيتون، يُنصح بالتدخل السريع.",
                "description_en": "Moderate Aculus Olearius infestation. Prompt action is recommended.",
                "severity_ar": "متوسط", "severity_en": "Moderate",
                "urgency_ar":  "قريباً", "urgency_en":  "Soon",
                "recommendations_ar": [
                    "رش مبيد أكاروسي (زيت معدني أو كبريت)",
                    "تجنب الري الزائد",
                    "تأكد من التهوية الجيدة بين الأشجار",
                    "راقب الوضع يومياً لمدة أسبوعين",
                ],
                "recommendations_en": [
                    "Spray acaricide (mineral oil or sulfur)",
                    "Avoid excessive irrigation",
                    "Ensure good ventilation between trees",
                    "Monitor daily for two weeks",
                ],
            }
        else:
            return {
                "description_ar": "بوادر إصابة خفيفة بأكاروس الزيتون، المراقبة كافية.",
                "description_en": "Early signs of mild Aculus Olearius. Monitoring is sufficient.",
                "severity_ar": "خفيف",   "severity_en": "Mild",
                "urgency_ar":  "مراقبة", "urgency_en":  "Monitor",
                "recommendations_ar": [
                    "راقب الورقة يومياً",
                    "رش محلول كبريتي وقائي خفيف",
                    "تجنب الإفراط في الري",
                ],
                "recommendations_en": [
                    "Monitor the leaf daily",
                    "Apply a light preventive sulfur solution",
                    "Avoid over-irrigation",
                ],
            }

    if class_key == "olive_knot":
        if confidence >= 90:
            return {
                "description_ar": "إصابة شديدة بعقدة الزيتون (البكتيريا)، يُنصح بالتدخل الفوري.",
                "description_en": "Severe Olive Knot infection. Immediate action is recommended.",
                "severity_ar": "شديد",      "severity_en": "Severe",
                "urgency_ar":  "عاجل جداً", "urgency_en":  "Urgent",
                "recommendations_ar": [
                    "قطع الفروع المصابة بمقص معقم وأحرقها فوراً",
                    "رش مبيد بكتيري نحاسي على الجروح والفروع",
                    "تعقيم أدوات التقليم بعد كل قطع",
                ],
                "recommendations_en": [
                    "Prune infected branches with sterilised tools and burn them immediately",
                    "Apply copper bactericide on wounds and branches",
                    "Sterilise pruning tools after each cut",
                    "Consult a specialist agricultural engineer",
                ],
            }
        elif confidence >= 70:
            return {
                "description_ar": "إصابة متوسطة بعقدة الزيتون، يُنصح بالتدخل السريع.",
                "description_en": "Moderate Olive Knot infection. Prompt action is recommended.",
                "severity_ar": "متوسط", "severity_en": "Moderate",
                "urgency_ar":  "قريباً", "urgency_en":  "Soon",
                "recommendations_ar": [
                    "قطع الفروع المصابة وتعقيم الأدوات",
                    "رش محلول نحاسي على مناطق الإصابة",
                    "تجنب إصابة الشجرة أثناء التقليم",
                ],
                "recommendations_en": [
                    "Prune infected branches and sterilise tools",
                    "Apply copper solution to affected areas",
                    "Avoid wounding the tree during pruning",
                ],
            }
        else:
            return {
                "description_ar": "بوادر إصابة خفيفة بعقدة الزيتون، المراقبة كافية.",
                "description_en": "Early signs of Olive Knot. Monitoring is sufficient.",
                "severity_ar": "خفيف",   "severity_en": "Mild",
                "urgency_ar":  "مراقبة", "urgency_en":  "Monitor",
                "recommendations_ar": [
                    "راقب الفروع بحثاً عن عقد أو انتفاخات",
                    "رش محلول نحاسي وقائي",
                    "تجنب التقليم في الطقس الممطر",
                ],
                "recommendations_en": [
                    "Monitor branches for knots or swellings",
                    "Apply preventive copper solution",
                    "Avoid pruning in rainy weather",
                ],
            }

    if class_key == "olive_peacock_spot":
        if confidence >= 90:
            return {
                "description_ar": "إصابة شديدة بمرض عين الطاووس، يُنصح بالتدخل الفوري.",
                "description_en": "Severe Peacock Spot infection. Immediate action is recommended.",
                "severity_ar": "شديد",      "severity_en": "Severe",
                "urgency_ar":  "عاجل جداً", "urgency_en":  "Urgent",
                "recommendations_ar": [
                    "رش مبيد فطري نحاسي فوراً (هيدروكسيد النحاس)",
                    "قطع وإتلاف جميع الأوراق والفروع المصابة",
                    "عزل الشجرة المصابة عن باقي الأشجار",
                    "تخلص من الأوراق الساقطة بالحرق",
                ],
                "recommendations_en": [
                    "Spray copper fungicide immediately (copper hydroxide)",
                    "Prune and destroy all infected leaves and branches",
                    "Isolate the infected tree from others",
                    "Dispose of fallen leaves by burning",
                ],
            }
        elif confidence >= 70:
            return {
                "description_ar": "إصابة متوسطة بعين الطاووس، يُنصح بالتدخل السريع.",
                "description_en": "Moderate Peacock Spot infection. Prompt action is recommended.",
                "severity_ar": "متوسط", "severity_en": "Moderate",
                "urgency_ar":  "قريباً", "urgency_en":  "Soon",
                "recommendations_ar": [
                    "رش مبيد فطري نحاسي أو كبريتي",
                    "إزالة الأوراق المصابة",
                    "تجنب الري الزائد",
                    "راقب الوضع أسبوعياً",
                ],
                "recommendations_en": [
                    "Spray copper or sulfur fungicide",
                    "Remove infected leaves",
                    "Avoid excessive irrigation",
                    "Monitor weekly",
                ],
            }
        else:
            return {
                "description_ar": "بوادر إصابة خفيفة بعين الطاووس، المراقبة والوقاية كافيتان.",
                "description_en": "Early signs of Peacock Spot. Monitoring and prevention are sufficient.",
                "severity_ar": "خفيف",   "severity_en": "Mild",
                "urgency_ar":  "مراقبة", "urgency_en":  "Monitor",
                "recommendations_ar": [
                    "راقب الورقة يومياً",
                    "رش محلول نحاسي وقائي خفيف",
                    "تأكد من وجود تهوية كافية حول الشجرة",
                ],
                "recommendations_en": [
                    "Monitor the leaf daily",
                    "Apply a light preventive copper solution",
                    "Ensure adequate ventilation around the tree",
                ],
            }

    return {
        "description_ar": "فئة غير معروفة.",
        "description_en": "Unknown class.",
        "severity_ar": "غير محدد", "severity_en": "Unknown",
        "urgency_ar":  "غير محدد", "urgency_en":  "Unknown",
        "recommendations_ar": [],
        "recommendations_en": [],
    }


# ============================================================
# Preprocessing
# ============================================================
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


# ============================================================
# فحص الصورة — Confidence + Entropy
# ============================================================
MIN_CONFIDENCE = 60.0
MAX_ENTROPY    = 0.85

def is_valid_olive_leaf(scores: list) -> tuple[bool, str]:
    probs          = np.array(scores)
    entropy        = float(-np.sum(probs * np.log(probs + 1e-10)))
    max_entropy    = np.log(len(scores))
    norm_entropy   = entropy / max_entropy
    max_confidence = max(scores) * 100

    if max_confidence < MIN_CONFIDENCE:
        return False, f"confidence too low ({max_confidence:.1f}%)"
    if norm_entropy > MAX_ENTROPY:
        return False, f"entropy too high ({norm_entropy:.2f})"
    return True, "ok"


# ============================================================
# Predict endpoint
# ============================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" in request.files:
            image_bytes = request.files["image"].read()
        elif request.json and "image" in request.json:
            image_bytes = base64.b64decode(request.json["image"])
        else:
            return jsonify({"error": "No image provided"}), 400

        input_data = preprocess_image(image_bytes)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        print(f"Raw output: {output_data}")

        scores      = [float(s) for s in output_data]
        class_index = int(np.argmax(scores))
        confidence  = round(scores[class_index] * 100, 2)

        valid, reason = is_valid_olive_leaf(scores)
        if not valid:
            print(f"❌ صورة مرفوضة: {reason}")
            return jsonify({
                "error":      "not_olive_leaf",
                "message_ar": "هذه الصورة لا تبدو ورقة زيتون. يرجى التقاط صورة واضحة لورقة زيتون.",
                "message_en": "This image does not appear to be an olive leaf. Please take a clear photo of an olive leaf.",
                "reason":     reason,
            }), 422

        class_key = CLASS_INDEX_TO_KEY[class_index]
        display   = CLASS_DISPLAY.get(class_key, {"ar": class_key, "en": class_key, "is_diseased": True})
        recs      = get_recommendations(class_key, confidence)

        print(f"Class: {class_index} ({class_key}), Confidence: {confidence}%")

        scores_dict = {
            CLASS_INDEX_TO_KEY[i]: round(scores[i] * 100, 2)
            for i in range(len(scores))
        }

        return jsonify({
            "class_index":        class_index,
            "class_key":          class_key,
            "label_ar":           display["ar"],
            "label_en":           display["en"],
            "confidence":         confidence,
            "is_diseased":        display["is_diseased"],
            "scores":             scores_dict,
            "description_ar":     recs["description_ar"],
            "description_en":     recs["description_en"],
            "severity_ar":        recs["severity_ar"],
            "severity_en":        recs["severity_en"],
            "urgency_ar":         recs["urgency_ar"],
            "urgency_en":         recs["urgency_en"],
            "recommendations_ar": recs["recommendations_ar"],
            "recommendations_en": recs["recommendations_en"],
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "classes": CLASS_INDEX_TO_KEY,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)