import ray

if ray.is_initialized():
    ray.shutdown()

ray.init(
    include_dashboard=False,  # disable dashboard that breaks on Windows
    ignore_reinit_error=True
)

@ray.remote
def hello():
    return "✅ Ray is working!"

print(ray.get(hello.remote()))
