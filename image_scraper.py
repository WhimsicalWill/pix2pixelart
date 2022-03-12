# scrape images using google-images-downloader package
from google_images_download import google_images_download

chromedriver_str = r"/usr/bin/chromedriver"
prompt = "Aesthetic pixel art landscape"
response = google_images_download.googleimagesdownload()
arguments = {"keywords": prompt, "limit": "5000", "print_urls": False, "chromedriver": chromedriver_str, "size": ">640*480"}
paths = response.download(arguments)
print(paths)