from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
from urllib.parse import urljoin
import re
import json

##dbs星展, sinopac永豐
bank_urls = {
            'dbs':'https://www.dbs.com.tw/personal-zh/cards/dbs-credit-cards/default.page', 
            'sinopac':'https://bank.sinopac.com/sinopacBT/personal/credit-card/introduction/list.html',
            'cathy': 'https://www.cathaybk.com.tw/cathaybk/personal/product/credit-card/cards/',
            'firstbank': 'https://card.firstbank.com.tw/sites/card/CreditCardList'
             }


credit_card_urls = {}
soup_documents = {k:None for k in list(bank_urls.keys())}

for bank, url in bank_urls.items():
    loader = RecursiveUrlLoader(url=url)
    docs = loader.load()
    if bank == 'dbs':
        links = list(set([a['href'] for a in Soup(docs[0].page_content).find_all('a', href=True) if (a['href'].startswith('/personal-zh/cards')) and (a['href'].endswith('hyperlink'))]))
        credit_card_urls[bank] = [urljoin(url, i) for i in links]
    elif bank == 'sinopac':
        links = list(set([a['href'] for a in Soup(docs[0].page_content).find_all('a', href=True) if a['href'].startswith('./')]))
        credit_card_urls[bank] = [urljoin(url, i) for i in links]
    elif bank == 'firstbank':
        def firstbank_extractor(html:str):
            soup_obj = Soup(html, 'html.parser')
            credit_card_feature = [re.sub(r'\n+', '\n', d.text) for d in soup_obj.find_all('div', {"class":[["card-single-features"]]})]
            txt = ''
            for features in  credit_card_feature:
                split = features.split('\n', 1)
                txt += f"信用卡卡名：{split[0].strip()}\n信用卡特色: {split[1]}"
            return txt
        all_html = RecursiveUrlLoader(bank_urls['firstbank'], extractor=firstbank_extractor).load()
        all_html[0].metadata['bank'] = 'firstbank'
        soup_documents[bank] = all_html

        divs = [Soup(doc.page_content, 'html.parser').find_all('div', {"class":'card-single'}) for doc in docs][0]
        links = []
        for d in divs:
            link_tags = d.find_all('a')
            if len(link_tags) == 1:
                detail_url = link_tags[0]['href']
                links.append(detail_url)
            else:
                for tag in link_tags:
                    if tag.text == '詳細內容':
                        links.append(tag['href'])
                    else:
                        continue
        credit_card_urls[bank] = [urljoin(url, i) for i in links]
    elif bank == 'cathy':
        def parser(html):
            ## main page
            divs = Soup(html, 'html.parser').find_all('div', {'class':'cubre-m-compareCard -credit'})
            if divs:
                divs = [re.sub(r'\n+', '\n', i.text) for i in divs]
                txt = (' ').join(divs)
                txt = re.sub(r' +', ' ', txt)
                txt = [t.replace('\n立即申辦','', 1) for t in txt.split('詳細說明') if t.startswith('\n立即申辦')]
                txt = ('').join(txt).replace('\n \n', '\n')
            else:
                ## deeper page
                divs = Soup(html, 'html.parser').find_all('div', class_=["cubre-o-textContent", "cubre-m-colorBanner__title","cubre-m-iconEssay__title","cubre-m-horGraphic__title","cubre-m-remind__title","cubre-m-puzzle__title","cubre-a-kvTitle -card"])
                divs = [d.text for d in divs if '您將離開本行官網 前往外部網站' not in d.text]
                uni_divs = []
                for d in divs[:len(divs)-1]:
                    if d not in uni_divs:
                        uni_divs.append(d)
                txt = re.sub(r'\n+', '\n', ('\n').join(uni_divs))
                txt = re.sub(r' +', ' ', txt)
            return txt
            
        docs = RecursiveUrlLoader(url=url, extractor=parser).load()
        for d in docs:
            d.metadata['bank'] = '國泰銀行'
            d.metadata['credit card'] = d.metadata['title'].split('-', 1)[0].strip()
            d.metadata['electronic payment']  = ''
        docs = [d for d in docs if '停發' not in d.metadata['title']]
        soup_documents['cathy'] = docs
        
def sinopac_extractor(html: str) -> str:
    soup = Soup(html, "html.parser")
    divs_txt = [s.text for s in soup.find_all('div', {'class':'tab-box'})]
    divs_txt.insert(0, re.sub(r'\n\n+', '\n', soup.find('div', {'class':'info'}).text))
    div_set = []
    for txt in divs_txt:
        if txt not in div_set:
            div_set.append(txt)
    txt = re.sub(r'[\t\r\xa0]', '', ('\n').join(div_set))
    txt = re.sub(r'  +', ' ', txt)
    txt = re.sub(r'\n+', '\n', txt)
    return txt.strip()

def dbs_extractor(html:str)->str:
    soup = Soup(html, "html.parser")
    divs_txt = [s.text for s in soup.find_all('div', {'class':'flpweb-legacy'})]
    if divs_txt:
        div_set = []
        for txt in divs_txt:
            if txt not in div_set:
                div_set.append(txt)
        all_txt = ('\n').join(div_set)
    else:
        txt = soup.text
        txt = [t.strip() for t in txt.split('\n') if( t != 'more..') and (len(txt.strip()) != 1)]
        all_txt = ('\n').join(txt)

    all_txt = all_txt.replace('個人網路銀行\nCard+ 信用卡數位服務\n企業網路銀行', '')
    all_txt = re.sub(r'[\t\r\xa0]', '', all_txt)
    all_txt = re.sub(r'\xa0', '', all_txt)
    all_txt = re.sub(r'  +', ' ', all_txt)
    all_txt = re.sub(r'\n+', '\n', all_txt)
    all_txt = all_txt.split('信用卡刷卡優惠更多星展信用卡刷卡優惠信息，詳情請見刷卡優惠說明。')[0]

    if '假字假字假字假字假字假字假字假字假字假字' in all_txt:
        all_txt = re.sub('假字假字假字+', '', all_txt)

    pattern = r"網銀登入\n[\s\S]*?選擇您的網站 ▼\n個人金融\n個人金融\n財富管理\n星展豐盛理財\n星展豐盛私人客戶\n企業金融\n中小企業銀行\n企業及機構銀行\n星展集團\n關於我們"
    # Replace the matched portion with an empty string
    all_txt = re.sub(pattern, '', all_txt)
    pattern = r'Loading...\n尊享至上\n[\s\S]*?\n權益手冊\n更多獨享優惠'
    all_txt = re.sub(pattern, '', all_txt)
    pattern = r'新戶好禮\n[\s\S]*?\n更多星展信用卡'
    all_txt = re.sub(pattern, '', all_txt)
    all_txt = all_txt.split('謹慎理財 信用至上')[0]
    return all_txt.strip()

def firstbank_extractor(html:str)->str:
    divs = Soup(html, 'html.parser').find_all('div', {"class":['carousel-item', 'content-body']})
    return ('').join([re.sub(r'\n+', '\n', d.text) for d in divs])

for bank, links in credit_card_urls.items():
    docs = []
    if bank == 'sinopac':
        for url in links:
            doc = RecursiveUrlLoader(url=url, extractor=sinopac_extractor).load()
            doc[0].metadata['bank'] = '永豐銀行'
            docs.extend(doc)
    elif bank == 'dbs':
        for url in links:
            doc = RecursiveUrlLoader(url=url, extractor=dbs_extractor).load()
            if doc:
                doc[0].metadata['bank'] = '星展銀行'
                docs.extend(doc)
    elif bank == 'firstbank':
        for url in links:
            doc = RecursiveUrlLoader(url=url, extractor=firstbank_extractor).load()
            doc[0].metadata['bank'] = '第一銀行'
            docs.extend(doc)
    if soup_documents[bank]:
        soup_documents[bank].extend(docs)
    else:
        soup_documents[bank] = docs

chinese_punctuation = "。！；：，、【】《》『』——"
translator = str.maketrans('', '', chinese_punctuation)

for k in list(soup_documents.keys()):
    if (k == 'sinopac'):
        for index, doc in enumerate(soup_documents[k]):
            credit_card = doc.page_content.split('\n')[0]
            # print(credit_card)
            if credit_card.endswith('卡'):
                soup_documents[k][index].metadata['credit card'] = credit_card
                soup_documents[k][index].metadata['electronic payment']  = ''
            elif 'pay' in credit_card.lower():
                soup_documents[k][index].metadata['credit card'] = ''
                soup_documents[k][index].metadata['electronic payment']  = credit_card
                
    elif k == 'dbs':
        for index, doc in enumerate(soup_documents[k]):
            credit_card = doc.page_content.split('\n')[0].split('|')[0].translate(translator)
            match = re.search(r'星展.*?卡', credit_card)
            if match:
                soup_documents[k][index].metadata['credit card'] = match.group()  # Output: 星展HAPPY GO聯名卡
                soup_documents[k][index].metadata['electronic payment']  = ''
            else:
                match = re.search(r'星展.*?卡', doc.metadata['description'])
                if match:
                    soup_documents[k][index].metadata['credit card'] = match.group()
                    soup_documents[k][index].metadata['electronic payment']  = ''
                else:
                    soup_documents[k][index].metadata['credit card'] = ''
                    pass
    elif k == 'firstbank':
        for index, doc in enumerate(soup_documents[k]):
            credit_card = doc.metadata['title'].strip()
            match = re.search(r'^(.*?)\s*-\s*信用卡', credit_card)
            if match:
                credit_card = match.group(1).split('_')[0]
                if credit_card.endswith('卡'):
                    soup_documents[k][index].metadata['credit card']  = credit_card
                    soup_documents[k][index].metadata['electronic payment']  = ''
                elif 'pay' in credit_card.lower():
                    soup_documents[k][index].metadata['credit card']  = ''
                    soup_documents[k][index].metadata['electronic payment']  = credit_card
                else:
                    soup_documents[k][index].metadata['credit card']  = ''
            else:
                pass

json_doc = {}
for key in list(soup_documents.keys()):
    docs = []
    for doc in soup_documents[key]:
        data = doc.metadata
        # del data['language']
        # del data['content_type']
        page_content = re.sub(r'\n\n+', '\n', doc.page_content).strip()
        page_content = re.sub(r't\t+', '\t', page_content)
        page_content = re.sub(r' +', ' ', page_content)
        data['page_content'] = page_content
        docs.append(data)
    json_doc[key] = docs


with open("/Users/sarah/Desktop/AI/credit_card_rag/documents.json", "w", encoding='utf8') as outfile: 
    json.dump(json_doc, outfile, ensure_ascii=False)

