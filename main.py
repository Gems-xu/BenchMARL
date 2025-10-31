def main():
    print("Hello from Gems' BenchMARL!")
    print("uv run python -m benchmarl.run algorithm=mappo task=vmas/navigation experiment.train_device='cuda' experiment.render=False")
    # experiment.max_n_frames=1_000_000
    # 或者使用迭代次数: experiment.max_n_iters=500


if __name__ == "__main__":
    main()
