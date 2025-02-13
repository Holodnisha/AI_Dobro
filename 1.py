def add_unique_words_to_set_from_string(input_string):
    unique_words = set()  # Создаем пустое множество для уникальных слов
    words = input_string.split()  # Разбиваем строку на слова
    for word in words:
        unique_words.add(word)  # Добавляем каждое слово в множество
    return unique_words

# Пример использования
input_string = "Это пример текста с уникальными словами пример текста"
unique_words_set = add_unique_words_to_set_from_string(input_string)
print(unique_words_set)