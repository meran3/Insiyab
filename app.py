"""
Insiyab: Crowd Density Estimation & Early Warning (Prototype)
Runs the trained crowd model and shows the results for each camera/zone.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

IMG_SIZE = 128            # Old model (Mall dataset)
DEN_H, DEN_W = 384, 512   # Density map model (ShanghaiTech)
DEN_SCALE = 100.0         # Same SCALE as in the training notebook
VGG_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)
APP_DIR = Path(__file__).parent
# The new density map model is tried first
MODEL_CANDIDATES = ["insiyab_density.keras", "best_model.keras", "crowd_cnn_model.h5"]

st.set_page_config(page_title="انسياب | كثافة الحشود", page_icon="🕋", layout="wide")

# Right-to-left page layout for Arabic
st.markdown(
    """
    <style>
    .stApp, .stMarkdown, .stMetric, [data-testid="stSidebar"] { direction: rtl; text-align: right; }
    .zone-card { border-radius: 12px; padding: 10px 14px; margin-bottom: 8px; color: #fff; font-weight: 600; }
    [data-testid="stVegaLiteChart"], .vega-embed, .vega-embed canvas { direction: ltr; }
    .low { background: #1f9d55; } .mid { background: #d69e2e; } .high { background: #c53030; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load the saved model from the app folder."""
    import tensorflow as tf

    for name in MODEL_CANDIDATES:
        path = APP_DIR / name
        if path.exists():
            return tf.keras.models.load_model(path, compile=False), name
    return None, None


def is_density_model(model) -> bool:
    return len(model.output_shape) == 4   # Outputs a map instead of a single number


def read_image(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def predict(model, bgr, rgb):
    """Return (count, density map or None) using the same preprocessing as training."""
    if is_density_model(model):
        x = cv2.resize(rgb, (DEN_W, DEN_H)).astype(np.float32)
        x = x[..., ::-1] - VGG_MEAN                     # RGB to BGR, then subtract VGG mean
        den = model.predict(x[None], verbose=0)[0, ..., 0] / DEN_SCALE
        return max(0.0, float(den.sum())), den
    x = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE)) / 255.0   # Old model: 128x128, divided by 255
    return max(0.0, float(model.predict(x[None], verbose=0)[0][0])), None


def density_maps(model, rgb_tiles):
    """Run the density model on a list of RGB images and return their density maps."""
    x = np.stack([cv2.resize(t, (DEN_W, DEN_H)).astype(np.float32)[..., ::-1] - VGG_MEAN for t in rgb_tiles])
    # Small batches keep memory low on the free Streamlit Cloud server
    return model.predict(x, batch_size=2, verbose=0)[..., 0] / DEN_SCALE


def predict_tiled(model, rgb, n):
    """High-resolution analysis (tiling): split the image into n x n tiles,
    analyze each tile at full model resolution, then add up the counts."""
    h, w = rgb.shape[:2]
    ys = np.linspace(0, h, n + 1).astype(int)
    xs = np.linspace(0, w, n + 1).astype(int)
    boxes = [(ys[i], ys[i + 1], xs[j], xs[j + 1]) for i in range(n) for j in range(n)]
    dens = density_maps(model, [rgb[y0:y1, x0:x1] for y0, y1, x0, x1 in boxes])

    # Stitch the tile maps back into one full-size map for the overlay
    full_map = np.zeros((h, w), dtype=np.float32)
    for (y0, y1, x0, x1), d in zip(boxes, dens):
        full_map[y0:y1, x0:x1] = cv2.resize(d, (x1 - x0, y1 - y0))
    return max(0.0, float(dens.sum())), full_map


def overlay(rgb, den):
    d = cv2.resize(den, (rgb.shape[1], rgb.shape[0]))
    d = d / (d.max() + 1e-9)
    heat = cv2.cvtColor(cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return (0.55 * rgb + 0.45 * heat).astype(np.uint8)


def density_level(count: float, mid: int, high: int):
    if count >= high:
        return "عالية", "high", "🔴"
    if count >= mid:
        return "متوسطة", "mid", "🟡"
    return "منخفضة", "low", "🟢"


# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ الإعدادات")
mid_th = st.sidebar.number_input("حد الكثافة المتوسطة (عدد الأشخاص)", min_value=1, value=300, step=10)
high_th = st.sidebar.number_input("حد الكثافة العالية (تنبيه)", min_value=2, value=600, step=10)
if high_th <= mid_th:
    st.sidebar.warning("حد الكثافة العالية لازم يكون أكبر من المتوسطة.")
st.sidebar.markdown("---")
st.sidebar.caption(
    "النظام أداة مساندة للقرار: يقدّر عدد الأشخاص وينبّه مبكراً، "
    "والقرار النهائي لفريق إدارة الحشود."
)

# ---------------- Main Page ----------------
st.title("انسياب")
st.subheader("نظام ذكي لقياس كثافة الحشود والتنبيه المبكر")

model, model_name = load_model()
if model is None:
    st.error(
        "ما لقيت ملف النموذج. ضعي الملف `best_model.keras` أو `crowd_cnn_model.h5` "
        "في نفس مجلد `app.py` ثم أعيدي تشغيل الصفحة."
    )
    st.stop()

density_mode = is_density_model(model)
show_map = st.sidebar.toggle("إظهار خريطة الكثافة", value=True) if density_mode else False
tiling = False
if density_mode:
    tiling = st.sidebar.toggle(
        "تحليل بدقة عالية (تقسيم الصورة)",
        value=False,
        help="يقسم الصورة إلى مربعات ويحلل كل مربع بدقته الكاملة. مناسب للقطات الواسعة جداً، وأبطأ قليلاً.",
    )
    grid = st.sidebar.select_slider("عدد المربعات", options=[2, 3, 4], value=3, format_func=lambda n: f"{n}×{n}", disabled=not tiling)

files = st.file_uploader(
    "ارفعي صورة أو أكثر — كل صورة تمثّل كاميرا في منطقة مختلفة",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if not files:
    st.info("ارفعي صور من الكاميرات لعرض الكثافة في كل منطقة.")
    st.stop()

results = []
with st.spinner("جاري تحليل الصور..."):
    for f in files:
        bgr, rgb = read_image(f.getvalue())
        if bgr is None:
            st.warning(f"تعذّر قراءة الصورة: {f.name}")
            continue
        count, den = predict(model, bgr, rgb)
        base_count = None
        if tiling:
            base_count = count                      # Keep the normal result for comparison
            count, den = predict_tiled(model, rgb, grid)
        zone = Path(f.name).stem
        level, css, icon = density_level(count, mid_th, high_th)
        results.append({"zone": zone, "count": count, "base": base_count, "level": level,
                        "css": css, "icon": icon, "img": rgb, "den": den})

if not results:
    st.stop()

# ---------- Overall Summary ----------
high_zones = [r for r in results if r["css"] == "high"]
c1, c2, c3 = st.columns(3)
c1.metric("عدد المناطق", len(results))
c2.metric("إجمالي الأشخاص (تقديري)", int(round(sum(r["count"] for r in results))))
c3.metric("مناطق بكثافة عالية", len(high_zones))

for r in high_zones:
    st.error(f"⚠️ تنبيه مبكر: كثافة عالية في «{r['zone']}» (≈ {r['count']:.0f} شخص). يُقترح توجيه الحركة لمنطقة أخف.")
if not high_zones:
    st.success("✔️ جميع المناطق ضمن الكثافة الطبيعية.")

# ---------- Zone Comparison Chart ----------
if len(results) > 1:   # A comparison needs at least two zones
    st.markdown("### مقارنة المناطق")
    import altair as alt

    chart_df = pd.DataFrame({
        "zone": [r["zone"] for r in results],
        "count": [round(r["count"]) for r in results],
        "level": [r["level"] for r in results],
    })
    # Horizontal bars, sorted from most crowded, colored like the zone cards
    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("count:Q", title="عدد الأشخاص"),
            y=alt.Y("zone:N", sort="-x", title=None),
            color=alt.Color(
                "level:N",
                scale=alt.Scale(domain=["منخفضة", "متوسطة", "عالية"], range=["#1f9d55", "#d69e2e", "#c53030"]),
                legend=None,
            ),
            tooltip=[alt.Tooltip("zone:N", title="المنطقة"), alt.Tooltip("count:Q", title="العدد"), alt.Tooltip("level:N", title="الكثافة")],
        )
        .properties(height=45 * len(chart_df) + 40)
    )
    st.altair_chart(chart)

# ---------- One Card per Camera ----------
st.markdown("### تفاصيل الكاميرات")
cols = st.columns(min(3, len(results)))
for i, r in enumerate(sorted(results, key=lambda r: -r["count"])):
    with cols[i % len(cols)]:
        if r["den"] is not None and show_map:
            st.image(overlay(r["img"], r["den"]), caption=f'{r["zone"]} — خريطة الكثافة')
        else:
            st.image(r["img"], caption=r["zone"])
        st.markdown(
            f'<div class="zone-card {r["css"]}">{r["icon"]} الكثافة: {r["level"]} — ≈ {r["count"]:.0f} شخص</div>',
            unsafe_allow_html=True,
        )
        if r["base"] is not None:
            st.caption(f"بدون تقسيم: ≈ {r['base']:.0f} شخص")

mode_note = f" — تحليل بدقة عالية ({grid}×{grid})" if tiling else ""
st.caption(f"النموذج المستخدم: {model_name}{mode_note} — الأعداد تقديرية.")
