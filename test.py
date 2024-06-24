from main import Main
from src.webhook import LarkHook

if __name__ == '__main__':
    main = Main()
    main.test()
    hook = LarkHook(url="https://www.feishu.cn/flow/api/trigger-webhook/b49c07c7d503dad13c8919f7ff71a707")
    data = {
        "is_finish": True
    }
    hook.send(data)
