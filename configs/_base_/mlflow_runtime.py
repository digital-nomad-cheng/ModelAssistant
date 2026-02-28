_base_ = ['./default_runtime.py']

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
        dict(type='MLflowVisBackend', save_dir='mlflow_runs')
    ]
)
