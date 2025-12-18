import yt_dlp
import os

def download_youtube(url: str, output_dir="downloads"):
    os.makedirs(output_dir, exist_ok=True)

    # ---- MP4 VIDEO ----
    video_opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": f"{output_dir}/%(title)s.%(ext)s",
        "merge_output_format": "mp4",
    }

    # ---- MP3 AUDIO ----
    audio_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{output_dir}/%(title)s.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }

    print("Downloading MP4...")
    with yt_dlp.YoutubeDL(video_opts) as ydl:
        ydl.download([url])

    print("Downloading MP3...")
    with yt_dlp.YoutubeDL(audio_opts) as ydl:
        ydl.download([url])

    print("✅ Download completed!")

if __name__ == "__main__":
    youtube_url = input("Enter YouTube video URL: ").strip()
    download_youtube(youtube_url)
