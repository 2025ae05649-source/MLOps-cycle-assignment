"""Draws the architecture diagram used in the report (reports/figures/architecture.png)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures",
                   "architecture.png")

boxes = {
    # name: (x, y, text, colour)
    "uci": (0.5, 6, "UCI repository\n(ucimlrepo id=45)", "#d9ead3"),
    "data": (3.5, 6, "download_data.py\nclean -> heart_clean.csv", "#d9ead3"),
    "eda": (6.5, 6, "EDA\n(notebook / eda.py)", "#d9ead3"),
    "train": (3.5, 4, "train.py\nLogReg + RandomForest\nGridSearchCV + 5-fold CV",
              "#cfe2f3"),
    "mlflow": (0.5, 4, "MLflow\nparams, metrics,\nplots, models", "#cfe2f3"),
    "model": (6.5, 4, "models/model.pkl\n(sklearn Pipeline)", "#cfe2f3"),
    "gh": (0.5, 2, "GitHub Actions\nlint -> train -> test\n-> docker build", "#fff2cc"),
    "docker": (3.5, 2, "Docker image\nFastAPI /predict", "#fff2cc"),
    "k8s": (6.5, 2, "Minikube (K8s)\nDeployment x2 +\nLoadBalancer Service", "#fff2cc"),
    "user": (9.5, 2, "User / curl\nJSON -> prediction\n+ confidence", "#eeeeee"),
    "prom": (6.5, 0, "Prometheus (docker compose)\nscrapes API /metrics", "#f4cccc"),
    "graf": (9.5, 0, "Grafana\ndashboard", "#f4cccc"),
}
arrows = [("uci", "data"), ("data", "eda"), ("data", "train"),
          ("train", "mlflow"), ("train", "model"), ("model", "docker"),
          ("gh", "docker"), ("docker", "k8s"), ("user", "k8s"),
          ("docker", "prom"), ("prom", "graf")]

W, H = 2.4, 1.2
fig, ax = plt.subplots(figsize=(13, 8))
for x, y, text, colour in boxes.values():
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.05",
                                facecolor=colour, edgecolor="black"))
    ax.text(x + W / 2, y + H / 2, text, ha="center", va="center", fontsize=9)


def centre(name):
    x, y = boxes[name][:2]
    return x + W / 2, y + H / 2


def edge_point(cx, cy, dx, dy):
    # point where a line from the box centre in direction (dx, dy) leaves the box
    hw, hh = W / 2 + 0.08, H / 2 + 0.08
    t = min(hw / abs(dx) if dx else 1e9, hh / abs(dy) if dy else 1e9)
    return cx + dx * t, cy + dy * t


for a, b in arrows:
    (x1, y1), (x2, y2) = centre(a), centre(b)
    dx, dy = x2 - x1, y2 - y1
    start = edge_point(x1, y1, dx, dy)
    end = edge_point(x2, y2, -dx, -dy)
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", lw=1.5))

ax.set_xlim(0, 12.5)
ax.set_ylim(-0.5, 7.7)
ax.axis("off")
ax.set_title("Heart disease MLOps pipeline - architecture", fontsize=14)
plt.tight_layout()
plt.savefig(OUT, dpi=120)
print("saved", OUT)
