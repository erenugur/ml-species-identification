from bs4 import BeautifulSoup
import requests

url = 'https://xeno-canto.org/species/Muscicapa-striata'
page = requests.get(url)
soup = BeautifulSoup(page.text, 'html')

table = soup.find('table', class_ = 'results') # this scrapes the section of the website that contains the downloadable audio files we want
print(table)

a_tags = table.find_all('a') # from the table variable, this scrapes all of the <a></a> tags (these contain all of the links)
print(a_tags)

all_links = [link.get("href") for link in a_tags] # this creates a list called "all_links" that contains all of the links in the <a></a> tags
print(all_links)

download_links = []
for i in all_links:           # this for loop takes all of the download links that we ACTUALLY want and puts it into a list
  if 'download' in i:
    download_links.append(i)
print(download_links)

for sub_url in download_links:
  r = requests.get(sub_url, allow_redirects=True)
  open(f'/content/drive/My Drive/AI Club Species Identification Project/Datasets/muscicapa_striata_audio/muscicapa_striata_audio{download_links.index(sub_url)}.wav', 'wb').write(r.content)
