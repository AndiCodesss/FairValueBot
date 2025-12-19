*** Settings ***
Documentation       Fair Value Finder Bot
...                 This robot automates the boring stuff! It goes to Yahoo Finance, scrapes the most active stocks,
...                 and then feeds them into my "Council of Five" AI models (including XGBoost and Deep Learning) to find hidden gems.
Library             RPA.Browser.Selenium
Library             RPA.Tables
Library             Collections
Library             String
Library             OperatingSystem
Library             AnalysisLibrary.py

*** Variables ***
${URL}              https://finance.yahoo.com/markets/stocks/most-active/?start=0&count=100

*** Tasks ***
Analyze Stocks with All Models
    Open Stock Screener
    Handle Cookies
    ${tickers}=    Scrape Tickers
    
    Log    Starting Analysis with Linear Regression...
    Run Linear Regression    ${tickers}
    
    Log    Starting Analysis with Random Forest...
    Run Random Forest    ${tickers}
    
    Log    Starting Analysis with Neural Network...
    Run Neural Network    ${tickers}

    Log    Starting Analysis with XGBoost...
    Run XGBoost    ${tickers}

    Log    Starting Analysis with Deep Learning (PyTorch)...
    Run Deep Learning    ${tickers}
    
    [Teardown]    Close All Browsers

*** Keywords ***
Open Stock Screener
    Open Available Browser    ${URL}
    Maximize Browser Window
    ${title}=    Get Title
    Log    Page Title: ${title}

Handle Cookies
    Run Keyword And Ignore Error    Wait Until Element Is Visible    xpath://button[@name='agree']    timeout=5s
    Run Keyword And Ignore Error    Click Element    xpath://button[@name='agree']
    Run Keyword And Ignore Error    Click Element    xpath://button[@value='agree']
    Run Keyword And Ignore Error    Click Element    xpath://button[contains(text(), 'Accept')]
    Run Keyword And Ignore Error    Click Element    xpath://button[contains(text(), 'Agree')]

Scrape Tickers
    Log    Scraping tickers from Yahoo Finance...
    Wait Until Element Is Visible    xpath://table    timeout=30s
    ${tickers}=    Create List
    ${rows}=    Get Element Count    xpath://table//tr
    Log To Console    Found ${rows} rows in the table.
    ${body_rows}=    Get Element Count    xpath://table/tbody/tr
    
    IF    ${body_rows} > 0
        FOR    ${i}    IN RANGE    1    ${body_rows} + 1
            ${ticker_element}=    Get WebElement    xpath://table/tbody/tr[${i}]/td[1]
            ${ticker}=    Get Text    ${ticker_element}
            ${ticker}=    Strip String    ${ticker}
            ${ticker}=    Fetch From Left    ${ticker}    ${SPACE}
            ${ticker}=    Replace String    ${ticker}    \n    ${EMPTY}
            Log    Found Ticker: ${ticker}
            Append To List    ${tickers}    ${ticker}
        END
    ELSE
        FOR    ${i}    IN RANGE    2    ${rows} + 1
            ${ticker_element}=    Get WebElement    xpath://table//tr[${i}]/td[1]
            ${ticker}=    Get Text    ${ticker_element}
            ${ticker}=    Strip String    ${ticker}
            ${ticker}=    Fetch From Left    ${ticker}    ${SPACE}
            ${ticker}=    Replace String    ${ticker}    \n    ${EMPTY}
            Log    Found Ticker: ${ticker}
            Append To List    ${tickers}    ${ticker}
        END
    END
    ${count}=    Get Length    ${tickers}
    Log To Console    Extracted ${count} tickers.
    RETURN    ${tickers}

