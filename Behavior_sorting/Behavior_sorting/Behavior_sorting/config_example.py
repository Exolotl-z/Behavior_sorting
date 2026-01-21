"""
鼠标行为标记系统 - 配置文件示例

这个文件展示了如何通过代码直接启动程序（适合批量处理）
"""

from enhanced_main import GroomingAnnotator

# 配置参数
config = {
    'video_path': r'D:\videos\mouse_video.avi',  # 视频文件路径
    'animal_id': 'M01',                          # 动物ID
    'session_id': 'baseline',                    # 会话ID
    'groom_key': 'g',                            # 标记按键
    'user_fps': 30,                              # 帧率，None为自动
}

# 启动程序
if __name__ == '__main__':
    # 方式1: 使用字典参数
    annotator = GroomingAnnotator(**config)
    annotator.run()
    
    # 方式2: 逐个指定参数
    # annotator = GroomingAnnotator(
    #     video_path=r'D:\videos\mouse_video.avi',
    #     animal_id='M01',
    #     session_id='baseline',
    #     groom_key='g',
    #     user_fps=30
    # )
    # annotator.run()
