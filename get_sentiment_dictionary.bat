@echo on

python sentiment_dictionary_multiprocessing.py --model md
python sentiment_dictionary_multiprocessing.py --model sm

timeout /t -1