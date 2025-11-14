  python examples/reproduce_table3_experiments.py --mode test

  运行单个数据集

  python examples/reproduce_table3_experiments.py --mode single --dataset MNIST

  运行全部实验（~500个实验）

  python examples/reproduce_table3_experiments.py --mode all

  查看结果

  mlflow ui --backend-store-uri experiments/table3_mlruns
  # 访问 http://localhost:5000