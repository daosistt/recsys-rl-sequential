"""Подготовить рисунки отчёта из сохранённых результатов, без обучения моделей.

Запуск из корня проекта: python -m report.build_figures.
"""

from src.config import Path, json, pd, plt, torch

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "artifacts/report_exports"
ASSETS = ROOT / "report/assets"
ASSETS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7})

LABELS = {"Baseline": "GPT-2", "GPT-2": "GPT-2", "Popularity": "Популярность",
          "Personal frequency": "Личные частоты", "RL listening": "RL: доля",
          "RL popularity_penalty": "RL: популярность", "RL repeat_penalty": "RL: повторы",
          "RL popularity penalty": "RL: популярность", "RL repeat penalty": "RL: повторы",
          "RL 1 step": "RL: 1 шаг", "RL 5 steps": "RL: 5 шагов"}


COLORS = {"GPT-2": "C0", "RL: доля": "C1", "RL: популярность": "C2",
          "RL: повторы": "C3", "RL: 1 шаг": "C3", "RL: 5 шагов": "C4"}

def save(fig: plt.Figure, filename: str) -> None:
    """Сохранить векторный рисунок и освободить фигуру."""
    fig.tight_layout(pad=0.8)
    fig.savefig(ASSETS / filename, bbox_inches="tight")
    plt.close(fig)


def step_plot(ax, frame: pd.DataFrame, column: str, dashed: bool = False) -> None:
    """Показать показатель по шагам для каждой модели."""
    for index, model in enumerate(frame.index.get_level_values("Model").unique()):
        values = frame.xs(model, level="Model")[column]
        ax.plot(values.index, values, marker=None if dashed else "o", markersize=3,
                linestyle="--" if dashed else "-", color=COLORS.get(LABELS.get(model, model), f"C{index}"),
                label=None if dashed else LABELS.get(model, model))
    ax.set_xlabel("Шаг")
    ax.set_xticks(range(1, 6))
    ax.grid(True, alpha=0.35)


# В exports история baseline пока хранится только на рисунке: берём числовую
# историю из того же исходного checkpoint через mmap, не создавая модель.
checkpoint = torch.load(ROOT / "artifacts/baseline_t4_1000_minfreq5/checkpoints/best.pt",
                        map_location="cpu", weights_only=True, mmap=True)
history = checkpoint["loss_history"]
frame = pd.DataFrame({"train": history["train_epoch"], "validation": history["val_epoch"]})
frame.index = range(1, len(frame) + 1)
frame.index.name = "epoch"
frame.to_csv(ASSETS / "baseline_history.csv")
del checkpoint
fig, ax = plt.subplots(figsize=(3.35, 2.15))
for key, label in (("train", "Train"), ("validation", "Validation")):
    ax.plot(frame.index, frame[key], marker="o", markersize=3, label=label, color=COLORS.get(label))
ax.set(xlabel="Эпоха", ylabel="Cross-entropy")
ax.set_xticks(frame.index)
ax.grid(True, alpha=0.35)
ax.legend()
save(fig, "baseline_loss.svg")

baseline = pd.read_csv(EXPORTS / "baseline_comparison.csv", index_col=["Model", "K"])
fig, ax = plt.subplots(figsize=(3.35, 2.15))
for model in baseline.index.get_level_values("Model").unique():
    row = baseline.xs(model)
    ax.plot(row.index, row["Recall@K"], marker="o", markersize=3, label=LABELS[model], color=COLORS.get(LABELS[model]))
ax.set(xlabel="K", ylabel="Recall@K")
ax.set_xticks([5, 10, 20])
ax.grid(True, alpha=0.35)
ax.legend()
save(fig, "baseline_comparison.svg")

reward = pd.read_csv(EXPORTS / "reward_comparison.csv", index_col=["Split", "Model"]).xs("test")
reward_labels = {"Train mean": "Среднее", "Track mean": "Среднее трека",
                 "GPT-2 + Ridge": "GPT-2 + Ridge", "GPT-2 + CatBoost": "GPT-2 + CB",
                 "GPT-2 + stats + CatBoost": "GPT-2 + 5 + CB", "Stats + CatBoost": "5 + CB"}
fig, ax = plt.subplots(figsize=(3.35, 2.3))
ax.barh([reward_labels[name] for name in reward.index], reward["MAE (п.п.)"])
ax.invert_yaxis()
ax.set_xlabel("Test MAE, п.п.")
ax.grid(True, axis="x", alpha=0.35)
save(fig, "reward_comparison.svg")

fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.3))
for name in ("listening", "popularity_penalty", "repeat_penalty"):
    data = pd.read_csv(ROOT / f"artifacts/rl_{name}_seed42/history.csv")
    label = LABELS[f"RL {name}"]
    axes[0].plot(data.epoch, data.val_ratio, marker="o", markersize=3, label=label, color=COLORS.get(label))
    axes[1].plot(data.epoch, data.val_kl, marker="o", markersize=3, label=label, color=COLORS.get(label))
for ax, ylabel in zip(axes, ("Validation predicted ratio", "Validation KL")):
    ax.set(xlabel="Эпоха", ylabel=ylabel)
    ax.set_xticks([0, 1, 2])
    ax.grid(True, alpha=0.35)
    ax.legend()
save(fig, "rl_training.svg")

rankings = pd.read_csv(EXPORTS / "rl_rewards_ranking.csv", index_col=["Model", "K"])
fig, ax = plt.subplots(figsize=(3.35, 2.25))
for model in rankings.index.get_level_values("Model").unique():
    data = rankings.xs(model)
    ax.plot(data.index, data["Recall@K"], marker="o", markersize=3, label=LABELS[model], color=COLORS.get(LABELS[model]))
ax.set(xlabel="K", ylabel="Recall@K")
ax.set_xticks([5, 10, 20])
ax.grid(True, alpha=0.35)
ax.legend()
save(fig, "rl_relevance.svg")

greedy = pd.read_csv(EXPORTS / "rollout_5step_greedy/step_metrics.csv", index_col=["Model", "Step"])
unique = pd.read_csv(EXPORTS / "rollout_5step_greedy_no_repeats/step_metrics.csv", index_col=["Model", "Step"])
fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.3))
for ax, column, ylabel in zip(axes, ("Predicted ratio", "Repeated in generation (%)"),
                              ("Прогноз доли прослушивания", "Повторы в генерации, %")):
    step_plot(ax, greedy, column)
    ax.set_ylabel(ylabel)
    ax.legend()
save(fig, "rollout_greedy.svg")

fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.3))
for ax, column, ylabel in zip(axes, ("Predicted ratio", "First-target hit@prefix (%)"),
                              ("Прогноз доли прослушивания", "Попадание цели в префикс, %")):
    step_plot(ax, unique, column)
    step_plot(ax, greedy, column, dashed=True)
    ax.set_ylabel(ylabel)
    ax.legend()
save(fig, "rollout_no_repeats.svg")

multi = pd.read_csv(EXPORTS / "evaluation_rl_5step_repeat_penalty_seed42/rollout_comparison.csv",
                    index_col=["Model", "Step"])
fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.3))
for ax, column, ylabel in zip(axes, ("Predicted ratio", "First-target hit@prefix (%)"),
                              ("Прогноз доли прослушивания", "Попадание цели в префикс, %")):
    step_plot(ax, multi, column)
    ax.set_ylabel(ylabel)
    ax.legend()
save(fig, "multistep_comparison.svg")

kl_behavior = pd.read_csv(EXPORTS / "kl_sweep_repeat_penalty_seed42/test_behavior_metrics.csv", index_col="Model")
kl_ranking = pd.read_csv(EXPORTS / "kl_sweep_repeat_penalty_seed42/test_ranking_metrics.csv",
                         index_col=["Model", "K"]).xs(20, level="K")
fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.3))
for name, row in kl_behavior.join(kl_ranking).iterrows():
    label = "GPT-2" if name == "Baseline" else name.replace("KL=", "β=")
    for ax, x, y in ((axes[0], row["KL"], row["Reward@1"]),
                     (axes[1], row["Reward@1"], row["NDCG@K"])):
        ax.scatter(x, y, s=15)
        ax.annotate(label, (x, y), xytext=(3, 4), textcoords="offset points", fontsize=7)
axes[0].set(xlabel="KL к исходной GPT-2", ylabel="Reward@1")
axes[1].set(xlabel="Reward@1", ylabel="NDCG@20")
for ax in axes:
    ax.margins(x=0.25, y=0.25)
    ax.grid(True, alpha=0.35)
axes[0].set_xlim(left=-0.004)
save(fig, "kl_tradeoff.svg")

with (ASSETS / "sources.json").open("w") as f:
    json.dump({"exports": "artifacts/report_exports", "baseline_history": "baseline_t4_1000_minfreq5/checkpoints/best.pt",
               "rl_histories": "artifacts/rl_<reward_type>_seed42/history.csv",
               "models_trained": False, "inference_run": False}, f, indent=2)
print(f"Сохранены 9 SVG в {ASSETS}. Обучение и inference не запускались.")


# Небольшие числовые таблицы доступны вместе с PDF, без весов моделей.
tables = ROOT / "report/tables"
tables.mkdir(parents=True, exist_ok=True)
for source, destination in {
    "data_summary.csv": "data_summary.csv",
    "baseline_comparison.csv": "baseline_comparison.csv",
    "reward_comparison.csv": "reward_comparison.csv",
    "reward_unfreezing_comparison.csv": "reward_unfreezing_comparison.csv",
    "rl_rewards_ranking.csv": "rl_rewards_ranking.csv",
    "rl_rewards_behavior.csv": "rl_rewards_behavior.csv",
    "rollout_5step_greedy/step_metrics.csv": "rollout_greedy.csv",
    "rollout_5step_greedy_no_repeats/step_metrics.csv": "rollout_no_repeats.csv",
    "evaluation_rl_5step_repeat_penalty_seed42/test_ranking_comparison.csv": "multistep_ranking.csv",
    "evaluation_rl_5step_repeat_penalty_seed42/rollout_comparison.csv": "multistep_rollout.csv",
    "kl_sweep_repeat_penalty_seed42/validation_summary.csv": "kl_validation.csv",
    "kl_sweep_repeat_penalty_seed42/test_ranking_metrics.csv": "kl_ranking.csv",
    "kl_sweep_repeat_penalty_seed42/test_behavior_metrics.csv": "kl_behavior.csv",
}.items():
    (tables / destination).write_bytes((EXPORTS / source).read_bytes())
(tables / "baseline_history.csv").write_bytes((ASSETS / "baseline_history.csv").read_bytes())
