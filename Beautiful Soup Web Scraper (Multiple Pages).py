from bs4 import BeautifulSoup
import requests

counter = 0

for x in range(8):
  file_num = 1
  url = f'https://xeno-canto.org/species/Muscicapa-striata?pg={x+35}'
  page = requests.get(url)
  soup = BeautifulSoup(page.text, 'html')

  table = soup.find('table', class_ = 'results') # this scrapes the section of the website that contains the downloadable audio files we want

  a_tags = table.find_all('a') # from the table variable, this scrapes all of the <a></a> tags (these contain all of the links)

  all_links = [link.get("href") for link in a_tags] # this creates a list called "all_links" that contains all of the links in the <a></a> tags

  download_links = []
  for i in all_links:           # this for loop takes all of the download links that we ACTUALLY want and puts it into a list
    if i is not None:
      if 'download.png' in i:
        continue
      if 'download' in i:
        download_links.append(i)

  for sub_url in download_links:
    r = requests.get(sub_url, allow_redirects=True)
    open(f'/content/drive/My Drive/AI Club Species Identification Project/Datasets/Test Data (200 files)/Audio/Muscicapa striata (200 audio files)/test{counter}.wav', 'wb').write(r.content)
    counter += 1
    print(f"File {x+35}.{file_num} Downloaded")
    file_num += 1

  print(f"--------------- Page {x+35} Completed")
