#!/usr/bin/python3

"""@author Hunter Hubers
   @Created 2019-04-03
"""

import scrape


def mass_build(file, url_list, value):
    """Function to bulk add training data

    :param file: Name of output file
    :param url_list: List of URLs to scrape
    :param value: label either 1 or 0
    :return: None
    """
    with open(file, "w+", encoding="utf-8") as OUT:
        for idx, url in enumerate(url_list):
            OUT.write(scrape.external_call(url)+","+str(value)+"\n")

    print('Done')


if __name__ == '__main__':

    urls = []
    filename = 'LeftDocs.txt'
    label = 1

    with open(filename, 'r', encoding='utf-8') as D:
        for line in D:
            urls.append(line.strip())

    mass_build("left.csv", urls, label)


