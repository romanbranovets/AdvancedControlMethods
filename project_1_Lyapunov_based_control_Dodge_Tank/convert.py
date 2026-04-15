import sys
import subprocess

def convert_mp4_to_gif(input_path, output_path, fps=10, width=320):
    """
    Конвертирует MP4 в GIF с помощью ffmpeg.
    """
    cmd = [
        'ffmpeg', '-i', input_path,
        '-vf', f'fps={fps},scale={width}:-1',
        '-loop', '0',
        output_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Готово: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Использование: python mp4_to_gif.py <входной.mp4> <выходной.gif> [fps] [ширина]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    width = int(sys.argv[4]) if len(sys.argv) > 4 else 320
    
    convert_mp4_to_gif(input_file, output_file, fps, width)