import time
import pandas as pd
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def get_attraction_url_list(driver_directory, base_url, num):

    options = Options()
    options.add_argument('--no-sandbox')
    options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(driver_directory, chrome_options=options)
    driver.maximize_window()
    driver.implicitly_wait(3)
    driver.get(base_url)

    see_more_class_name = \
        'attractions-attraction-overview-main-TopPOIs__see_more--2Vsb-'
    page_num_info_class_name = \
        'attractions-attraction-overview-main-Pagination__link--2m5mV'
    url_class1_name = \
        'attractions-attraction-overview-pois-PoiCard__item--3UzYK'
    in_page_class_name = 'attraction_list'
    url_class2_name = 'more'
    next_page_class_name = 'nav.next'

    driver.find_element_by_class_name(see_more_class_name).click()
    time.sleep(2)

    end_page = driver.find_elements_by_class_name(page_num_info_class_name)[-2]
    end_page = int(end_page.text)

    url_list = []
    for c_ in driver.find_elements_by_class_name(url_class1_name):
        c_ = c_.find_element_by_css_selector('div> div> div:nth-child(3)> a')
        url_list.append(c_.get_attribute('href'))

    driver.find_elements_by_class_name(page_num_info_class_name)[-1].click()
    page = 2

    while len(url_list) < num:

        if page > end_page:
            break

        while True:
            try:
                in_page = driver.find_element_by_class_name(in_page_class_name)
                url_element_list = \
                    in_page.find_elements_by_class_name(url_class2_name)
                url_list_page = []
                for c_ in url_element_list:
                    c_ = c_.find_element_by_css_selector('a')
                    url_list_page.append(c_.get_attribute('href'))
                page += 1
                driver.find_element_by_class_name(next_page_class_name).click()
            except ValueError:
                check_blackbox(driver)
                time.sleep(2)
                continue

            url_list.extend(url_list_page)

            break

    driver.close()

    return url_list[: num]


def check_blackbox(driver):

    if driver.find_elements_by_class_name('sbx_banner'):

        driver.find_element_by_class_name('sbx_close').click()


class review_collector():

    def __init__(self, driver_directory, url):

        chrome_options = Options()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument("start-maximized")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-setuid-sandbox")

        self.url_ = url
        self.driver = webdriver.Chrome(
            driver_directory, chrome_options=chrome_options)
        self.driver.maximize_window()
        self.driver.implicitly_wait(3)
        self.driver.get(self.url_)
        self.waiting_time_ = 1
        self.retry_ = 0

        self.month_list_ = [
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        self.current_year_ = time.ctime().split()[-1]
        self.current_month_ = time.ctime().split()[1]

        self.read_more_class_name_ = \
            'location-review-review-list-parts-ExpandableReview__cta--2mR2g'
        self.read_less_class_name_ = \
            'location-review-review-list-parts-ExpandableReview__'\
            'ctaCollapse--TGOvo'
        self.review_list_class_name_ = \
            'location-review-card-Card__ui_card--2Mri0'
        self.date_class_name_ = \
            'social-member-event-MemberEventOnObjectBlock__event_type--3njyv'
        self.title_class_name_ = \
            'location-review-review-list-parts-ReviewTitle__reviewTitle--2GO9Z'
        self.rating_class_name_ = \
            'location-review-review-list-parts-RatingLine__bubbles--GcJvM'
        self.review_class_name_ = \
            'location-review-review-list-parts-ExpandableReview__reviewText'\
            '--gOmRC'
        self.next_page_class_name_ = 'ui_pagination'
        self.select_eng_class_name_ = \
            'location-review-review-list-parts-LanguageFilter__no_wrap--2Dckv'
        self.location_class_name_ = \
            'location-review-location-review-layout-Header__headerText--2dAcj'
        self.number_of_review_class_name_ = \
            'location-review-review-list-parts-LanguageFilter__paren_count--'\
            '2vk3f'

        feature_list = [
            'attraction', 'date', 'writer', 'title', 'review', 'rating']
        self.dict_ = {feature: [] for feature in feature_list}

        self.pageNum_ = self.driver.find_element_by_class_name('pageNumbers')
        self.pageNum_ = self.pageNum_.find_elements_by_css_selector('a')
        self.pageNum_ = int(self.pageNum_.[-1].text)
        self.attraction_name_ = self.url_[
            self.url_.find('Reviews')+8: self.url_.find('-Seoul.html')]

        print('%s collect' % self.attraction_name_)

    def read_more(self):

        c_name = self.read_more_class_name_
        element = self.driver.find_element_by_class_name(c_name)
        element.click()

    def read_less(self):

        c_name = self.read_less_class_name_
        element = self.driver.find_elements_by_class_name(c_name)
        if not element:
            self.driver.find_element_by_class_name(c_name).click()

    def select_eng(self):

        while True:
            try:
                self.page_move()
                c_name = self.select_eng_class_name_
                langlist = self.driver.find_elements_by_class_name(c_name)
                eng = None
                for i, l in enumerate(langlist):
                    if l.text == 'English':
                        eng = l
                        self.language_idx_ = i

                self.retry_ = 0

                if eng is not None:
                    eng.click()
                    time.sleep(self.waiting_time_)
                    return True
                else:
                    return False
            except ValueError:
                time.sleep(self.waiting_time_)
                self.retry_ += 1
                if self.retry_ == 100:
                    return False

                continue

    def page_move(self):

        while True:
            try:
                c_name = self.location_class_name_
                element = self.driver.find_element_by_class_name(c_name)
                _, y_loc_ = element.location.values()
                self.driver.execute_script('window.scrollTo(0,%s);' % (y_loc_))
            except ValueError:
                time.sleep(self.waiting_time_)
                self.retry_ += 1
                if self.retry_ == 100:
                    self.refresh()
                continue

            self.retry_ = 0

            break

    def number_of_review(self):

        c_name = self.number_of_review_class_name_
        element = self.driver.find_elements_by_class_name(c_name)
        num = element[self.language_idx_ - 1].text
        num = re.sub(re.compile('[()]'), '', num).replace(' ', '')

        return int(num)

    def get_writer_info(self, element):

        info_ = element.find_element_by_class_name(self.date_class_name_).text

        writer_ = info_[:info_.find(' wrote')]
        date_info_ = info_.split()[-2:]

        if date_info_[0] not in self.month_list_:
            date_info_[0] = self.current_month_
        if not bool(re.search(r'\d{4}', date_info_[1])):
            date_info_[1] = self.current_year_

        date_ = ' '.join([str(d_) for d_ in date_info_])

        return writer_, date_

    def next_page(self):

        c_name = self.next_page_class_name_
        element = self.driver.find_element_by_class_name(c_name)
        element.find_elements_by_class_name('ui_button')[-1].click()

    def get_title(self, element):

        return element.find_element_by_class_name(self.title_class_name_).text

    def get_rating(self, element):

        element = element.find_element_by_class_name(self.rating_class_name_)
        element = element.find_element_by_css_selector('span')

        return int(element.get_attribute('class')[-2:])//10

    def get_review(self, element):

        return element.find_element_by_class_name(self.review_class_name_).text

    def get_info(self):

        while True:
            try:
                self.read_more()
            except ValueError:
                self.retry_ += 1
                if self.retry_ % 100 == 0:
                    self.read_less()
                time.sleep(self.waiting_time_)
                continue
            break

        time.sleep(self.waiting_time_)

        while True:
            try:

                attraction_list = []
                date_list = []
                writer_list = []
                title_list = []
                review_list = []
                rating_list = []
                name_ = review_list_class_name_

                for element in self.driver.find_elements_by_class_name(name_):
                    writer, date = self.get_writer_info(element)
                    attraction_list.append(self.attraction_name_)
                    date_list.append(date)
                    writer_list.append(writer)
                    title_list.append(self.get_title(element))
                    review_list.append(self.get_review(element))
                    rating_list.append(self.get_rating(element))

            except ValueError:
                self.retry_ += 1
                if self.retry_ % 10 == 0:
                    self.driver.refresh()
                    self.page_move()
                    self.read_more()
                time.sleep(self.waiting_time_)
                continue

            writer, date = self.get_writer_info(element)

            self.dict_['attraction'].extend(attraction_list)
            self.dict_['date'].extend(date_list)
            self.dict_['writer'].extend(writer_list)
            self.dict_['title'].extend(title_list)
            self.dict_['review'].extend(review_list)
            self.dict_['rating'].extend(rating_list)

            self.retry_ = 0

            break

    def get_df(self):

        df = pd.DataFrame(self.dict_)
        df.to_csv(
            '../reviews/%s.csv' % self.attraction_name_.replace(' ', '_'),
            sep=',',
            index=False)

    def execute(self):

        while True:
            if self.driver.current_url[-8:] != '#REVIEWS':
                self.url_ += '#REVIEWS'
                self.driver.get(self.url_)
                time.sleep(2)
                continue

            if not self.select_eng():
                break
            if self.number_of_review() == 0:
                break

            self.get_info()

            if self.number_of_review() <= 5:
                self.get_df()
                break
            for _ in range(self.pageNum_ - 1):
                self.next_page()
                self.get_info()
            self.get_df()
            break

        self.driver.close()


if __name__ == '__main__':

    driver_directory = '../chromedriver.exe'
    url = 'https://www.tripadvisor.co.za/Attraction_Review-g294197-d592495'\
        '-Reviews-Coex_Aquarium-Seoul.html#REVIEWS'

    collector = review_collector(driver_directory, url)
    collector.execute()
