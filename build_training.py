#!/usr/bin/python3

"""@author Hunter Hubers
   @Created 2019-04-03
"""

import ast
import re
import os

import scrape


def mass_join(folder):
    """Joins distinct training files into a corpus file.

       :param folder: The folder to loop through and join files in.
       :return: None
    """
    with open(folder+"/output.txt", "w+") as output:  # Create output file
        for text_file in os.listdir(folder):  # Loop through the folder
            if text_file.startswith("."):  # Skip hidden files
                continue
            with open(folder+text_file, 'r') as F:  # Open file
                for line in F:
                    for string in ast.literal_eval(line):  # Evaluate the data as a list
                        if len(string) < 15:  # Some text cleanup
                            continue
                        output.write(string+"\n")
                    break  # We only care about the first file


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


def run_bulk_scrape():
    """Bulk Scrape Web Pages"""
    s = scrape.WebScrape()

    urls = ['https://www.cnn.com/2019/04/03/us/ivy-league-college-admissions-trnd/index.html',
            'https://www.cnn.com/2019/04/03/politics/fbi-mar-a-lago-florida/index.html',
            'https://www.cnn.com/2019/04/01/politics/house-judiciary-subpoena-full-muller-report/index.html',
            'https://www.cnn.com/2019/04/01/politics/supreme-court-planned-parenthood-video-lawsuit/index.html']

    mass_build(s, urls)


if __name__ == '__main__':

    # Build training data file.
    mass_join('./right_wing/')
