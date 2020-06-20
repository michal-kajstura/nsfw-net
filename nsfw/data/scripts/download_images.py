from nsfw.data.download_sample_data import download_data

neutral_url = 'https://github.com/alex000kim/nsfw_data_scraper/raw/master/raw_data/neutral/urls_neutral.txt'
nsfw_url = 'https://github.com/alex000kim/nsfw_data_scraper/raw/master/raw_data/porn/urls_porn.txt'

neutral_path = '/home/michal/data/zpi/images/neutral'
nsfw_path = '/home/michal/data/zpi/images/nsfw'

download_data(neutral_url, neutral_path, 100)
# download_data(nsfw_url, nsfw_path, 50)
