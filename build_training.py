#!/usr/bin/python3

"""@author Hunter Hubers
   @Created 2019-04-03
"""

import re

import scrape


def mass_build(scraper, url_list):
    """Function to create training data in bulk

    :param scraper: An instantiated class from scrape.py
    :param url_list: List of URLs
    :return: None
    """
    for idx, url in enumerate(url_list):
        site = re.search(r'(?<=www.)(.*?)\w+(?=.)', url).group(0)
        # Pull the page data
        scraper.get_page(url)
        scraper.set_data()

        # Clean and export the data
        text_data = scraper.clean_content()
        scraper.export_data("./training_data/"+site+"/"+site+str(idx)+'.txt', text_data)
    print('Done')


if __name__ == '__main__':
    # Initialize the class
    S = scrape.WebScrape()
    urls = ['https://www.cnn.com/2019/04/03/us/ivy-league-college-admissions-trnd/index.html',
            'https://www.cnn.com/2019/04/03/politics/fbi-mar-a-lago-florida/index.html',
            'https://www.cnn.com/2019/04/01/politics/house-judiciary-subpoena-full-muller-report/index.html',
            'https://www.cnn.com/2019/04/01/politics/supreme-court-planned-parenthood-video-lawsuit/index.html']
    mass_build(S, urls)
