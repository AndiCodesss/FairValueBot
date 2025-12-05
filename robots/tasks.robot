*** Settings ***
Documentation       Fair Value Finder Bot
...                 This robot automates the boring stuff! It goes to Yahoo Finance, scrapes the most active stocks,
...                 and then feeds them into my "Council of Five" AI models (including XGBoost and Deep Learning) to find hidden gems.
Library             RPA.Browser.Selenium
Library             RPA.Tables
Library             Process
Library             Collections
Library             String
Library             OperatingSystem

*** Variables ***
${URL}              https://finance.yahoo.com/markets/stocks/most-active/?start=0&count=100
${PYTHON_EXE}       ${EXECDIR}${/}.venv${/}Scripts${/}python.exe

*** Tasks ***
Analyze Stocks with All Models
    Open Stock Screener
    Handle Cookies
    ${tickers}=    Scrape Tickers
    
    Log    Starting Analysis with Linear Regression...
    Analyze Tickers    ${tickers}    src${/}analysis${/}predict_lr.py    output_lr.txt
    
    Log    Starting Analysis with Random Forest...
    Analyze Tickers    ${tickers}    src${/}analysis${/}predict_rf.py    output_rf.txt
    
    Log    Starting Analysis with Neural Network...
    Analyze Tickers    ${tickers}    src${/}analysis${/}predict_ml.py    output_ml.txt

    Log    Starting Analysis with XGBoost...
    Analyze Tickers    ${tickers}    src${/}analysis${/}predict_xgb.py    output_xgb.txt

    Log    Starting Analysis with Deep Learning (PyTorch)...
    Analyze Tickers    ${tickers}    src${/}analysis${/}predict_dl.py    output_dl.txt
    
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

Analyze Tickers
    [Arguments]    ${tickers}    ${script_path}    ${output_file}
    Log    Running analysis script: ${script_path}
    
    ${result}=    Run Process    ${PYTHON_EXE}    ${EXECDIR}${/}${script_path}    @{tickers}
    ...    stdout=${EXECDIR}${/}${output_file}    stderr=STDOUT    timeout=10min
    
    Log    Script Output:
    ${output}=    Get File    ${EXECDIR}${/}${output_file}
    Log    ${output}
    
    RETURN    ${result.rc}
