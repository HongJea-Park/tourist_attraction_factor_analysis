import multiprocessing as mp
import argparse

from data_collection import tripadvisor_attraction


def main(driver_directory, url):

    collector = tripadvisor_attraction.review_collector(driver_directory, url)
    collector.execute()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_cpu', type=int, default=-1,
        help='The number of CPU for multiprocessing')

    args = parser.parse_args()

    if args.n_cpu == -1:
        n_cpu = mp.cpu_count()
    else:
        n_cpu = args.n_cpu

    driver_directory = '../chromedriver.exe'
    base_url = 'https://www.tripadvisor.co.za/'\
        'Attractions-g294197-Activities-Seoul.html'
    url_list = tac.get_attraction_url_list(
        driver_directory=driver_directory,
        base_url=base_url,
        num=100)

    url_list = [(driver_directory, url) for url in url_list]

    mp.freeze_support()
    with mp.Pool(processes=n_cpu) as pool:
        pool.starmap(main, url_list)
