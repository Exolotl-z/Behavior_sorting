"""
数据分析工具 - 独立版本

这个脚本可以单独运行，用于分析已经保存的CSV文件
无需重新打开视频，直接查看数据统计和可视化
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def analyze_csv(csv_file):
    """分析CSV文件并生成统计报告和图表"""
    
    if not os.path.exists(csv_file):
        print(f"错误: 文件 {csv_file} 不存在")
        return
    
    # 读取数据
    df = pd.read_csv(csv_file)
    print(f"\n已加载数据文件: {csv_file}")
    print(f"数据行数: {len(df)}")
    
    if len(df) == 0:
        print("警告: CSV文件中没有数据")
        return
    
    # 确保有必要的列
    required_cols = ['animal_id', 'session_id', 'start_s', 'end_s', 'duration_s']
    if not all(col in df.columns for col in required_cols):
        print(f"错误: CSV文件缺少必要的列。需要: {required_cols}")
        return
    
    # 提取信息
    animal_id = df['animal_id'].iloc[0]
    session_id = df['session_id'].iloc[0]
    
    # 计算统计量
    total_duration = df['duration_s'].sum()
    mean_duration = df['duration_s'].mean()
    median_duration = df['duration_s'].median()
    std_duration = df['duration_s'].std()
    count = len(df)
    min_duration = df['duration_s'].min()
    max_duration = df['duration_s'].max()
    
    # 生成文本报告
    print("\n" + "="*60)
    print("          行为标记统计分析报告")
    print("="*60)
    print(f"\n动物ID: {animal_id}")
    print(f"会话ID: {session_id}")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "-"*60)
    print("基本统计")
    print("-"*60)
    print(f"标记区间总数:        {count} 次")
    print(f"总持续时间:          {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
    print("\n" + "-"*60)
    print("持续时间统计")
    print("-"*60)
    print(f"平均持续时间:        {mean_duration:.2f} 秒")
    print(f"中位数持续时间:      {median_duration:.2f} 秒")
    print(f"标准差:              {std_duration:.2f} 秒")
    print(f"最短持续时间:        {min_duration:.2f} 秒")
    print(f"最长持续时间:        {max_duration:.2f} 秒")
    print("="*60 + "\n")
    
    # 创建可视化
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'行为分析报告 - {animal_id} ({session_id})', 
                 fontsize=16, fontweight='bold')
    
    # 1. 持续时间分布直方图
    ax1 = plt.subplot(2, 2, 1)
    ax1.hist(df['duration_s'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('持续时间 (秒)', fontsize=12)
    ax1.set_ylabel('频数', fontsize=12)
    ax1.set_title('持续时间分布直方图', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. 时间线
    ax2 = plt.subplot(2, 2, 2)
    for idx, row in df.iterrows():
        ax2.barh(0, row['duration_s'], left=row['start_s'], height=0.5,
                color='green', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('时间 (秒)', fontsize=12)
    ax2.set_yticks([])
    ax2.set_title('行为发生时间线', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. 箱线图
    ax3 = plt.subplot(2, 2, 3)
    bp = ax3.boxplot([df['duration_s']], vert=True, patch_artist=True,
                     labels=['持续时间'], widths=0.5)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['medians'][0].set_color('red')
    bp['medians'][0].set_linewidth(2)
    ax3.set_ylabel('持续时间 (秒)', fontsize=12)
    ax3.set_title('持续时间箱线图', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 累积时间曲线
    ax4 = plt.subplot(2, 2, 4)
    df_sorted = df.sort_values('start_s')
    cumulative_time = df_sorted['duration_s'].cumsum()
    ax4.plot(df_sorted['start_s'], cumulative_time, marker='o', 
            color='purple', linewidth=2, markersize=4)
    ax4.set_xlabel('时间点 (秒)', fontsize=12)
    ax4.set_ylabel('累积行为时长 (秒)', fontsize=12)
    ax4.set_title('累积行为时长曲线', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = f"{animal_id}_{session_id}_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {output_file}")
    
    # 显示图表
    plt.show()
    
    # 生成详细报告文件
    report_file = f"{animal_id}_{session_id}_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("          行为标记统计分析报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"动物ID: {animal_id}\n")
        f.write(f"会话ID: {session_id}\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("-"*60 + "\n")
        f.write("基本统计\n")
        f.write("-"*60 + "\n")
        f.write(f"标记区间总数:        {count} 次\n")
        f.write(f"总持续时间:          {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)\n\n")
        f.write("-"*60 + "\n")
        f.write("持续时间统计\n")
        f.write("-"*60 + "\n")
        f.write(f"平均持续时间:        {mean_duration:.2f} 秒\n")
        f.write(f"中位数持续时间:      {median_duration:.2f} 秒\n")
        f.write(f"标准差:              {std_duration:.2f} 秒\n")
        f.write(f"最短持续时间:        {min_duration:.2f} 秒\n")
        f.write(f"最长持续时间:        {max_duration:.2f} 秒\n")
        f.write("="*60 + "\n\n")
        f.write("详细数据:\n")
        f.write(df.to_string(index=False))
    
    print(f"详细报告已保存: {report_file}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # 从命令行参数获取文件名
        csv_file = sys.argv[1]
    else:
        # 交互式输入
        print("="*60)
        print("          数据分析工具")
        print("="*60)
        print("\n请输入CSV文件路径 (例如: M01_grooming.csv)")
        print("或直接拖动CSV文件到此窗口，然后按回车:\n")
        csv_file = input("> ").strip().strip('"')
    
    analyze_csv(csv_file)
    
    print("\n分析完成！按任意键退出...")
    input()
