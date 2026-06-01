import numpy as np
import matplotlib.pyplot as plt

def plot_training_curve(rewards, title="Training Curve"):
    plt.figure()

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)

    plt.show()

def plot_entropy_curve(entropy_values, title="Policy Entropy Over Time"):
    plt.figure()

    plt.plot(entropy_values)
    plt.xlabel("Training Step")
    plt.ylabel("Entropy")
    plt.title(title)

    plt.show()

def plot_before_after(before_map, after_map):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im1 = axes[0].imshow(before_map)
    axes[0].set_title("Before Cue Removal")

    im2 = axes[1].imshow(after_map)
    axes[1].set_title("After Cue Removal")

    plt.colorbar(im1, ax=axes[0])
    plt.colorbar(im2, ax=axes[1])

    plt.show()

def plot_uncertainty_vs_metric(uncertainty, metric, ylabel="Stabilization"):
    plt.figure()

    plt.plot(uncertainty, metric, marker="o")
    plt.xlabel("Uncertainty Level")
    plt.ylabel(ylabel)
    plt.title("Effect of Uncertainty on Behavioural Stabilization")

    plt.show()

def plot_cue_dependence(cue_scores):
    plt.figure()

    plt.plot(cue_scores)
    plt.xlabel("Run")
    plt.ylabel("Cue Dependence Score")
    plt.title("Cue Dependence Across Runs")

    plt.show()
