import json
from datetime import datetime


def convert_to_json(file_path, output_file_path):
    result_arr = []

    with open(file_path, 'r') as file:
        first_line = file.readline()
        titles = [title.strip() for title in first_line.split('\t')]

        for line in file:
            empty_dict = {}
            values = line.split("\t")
            if len(values) > 3:
                values[-1] = values[-1].split(",")
                values[-1] = list(map(lambda x: x.strip(), values[-1]))
                date_obj = datetime.strptime(values[-2], "%Y/%m/%d")
                values[-2] = date_obj.strftime("%B %d, %Y")
                for title, value in zip(titles, values):
                    if type(value) is str:
                        empty_dict[title] = value.strip()
                    else:
                        empty_dict[title] = value
                result_arr.append(empty_dict)

    json_data = json.dumps(result_arr, indent=4)

    with open(output_file_path, 'w', encoding='utf8') as output_file:
        output_file.write(json_data)


input_file = "output.txt"
output_file = "output.json"

convert_to_json(input_file, output_file)

# PMID	abstract	publication_date	authors
print(f"Conversion completed. JSON data saved to {output_file}")
