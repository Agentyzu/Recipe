import os
import urllib.request
import zipfile

def download_and_extract_wordnet():
    nltk_data_dir = os.path.expanduser('~/nltk_data/corpora')
    os.makedirs(nltk_data_dir, exist_ok=True)
    zip_path = os.path.join(nltk_data_dir, 'wordnet.zip')

    # Candidate sources for the zip file (primary and mirror)
    urls = [
        # GitHub
        "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip",
        

        "https://mirrors.tuna.tsinghua.edu.cn/nltk_data/packages/corpora/wordnet.zip",
    ]

    print("Starting WordNet download (automatic mirror switching)")
    print(f"Target path: {zip_path}")

    success = False

    for url in urls:
        try:
            print(f"\n Attempting: {url}")
            urllib.request.urlretrieve(url, zip_path)
            print(" Download successful!")
            success = True
            break
        except Exception as e:
            print(f"Failed: {e}")

    if not success:
        print("\n All download sources failed. Please check your network environment.")
        return

    # Extracting the zip file
    try:
        print("Extracting")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(nltk_data_dir)

        print("Extraction successful! WordNet is ready")
    except Exception as e:
        print(f"Extraction failed: {e}")


if __name__ == "__main__":
    download_and_extract_wordnet()