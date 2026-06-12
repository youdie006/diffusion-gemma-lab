# 논문 스타일 matplotlib 설정 (모든 실험 공용)
# 원칙: serif 폰트, 안쪽 틱, 상/우 스파인 제거, 무채색 + 강조색 1개,
#       그림 내부 제목 없음 (캡션은 README의 Figure N 텍스트로), 300dpi 저장

import matplotlib as mpl

INK = "#1a1a1a"       # 본문 검정
GRAY = "#9a9a9a"      # 보조 회색
ACCENT = "#3a5fa0"    # 강조 (남색)
ACCENT2 = "#a04040"   # 보조 강조 (벽돌색)
LIGHT = "#dfe6f0"     # 음영 영역


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": False,
    })


def panel_label(ax, label):
    """패널 좌상단에 (a), (b) 라벨."""
    ax.text(-0.08, 1.04, label, transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", va="bottom", ha="right")
