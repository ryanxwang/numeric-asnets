import argparse
import os

def adjust_prices(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract the base file name without the extension
    base_file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Sets to store markets, depots, and existing self drive-costs
    markets_depots = set()
    existing_self_drive_costs = set()

    # Dictionary to store the line numbers and indentation of goods with on-sale 0
    goods_not_on_sale = {}

    # Set to store goods with specified prices
    goods_with_price = set()

    # Hardcoded indentation of the price lines
    indentation = 1

    # Index of the first drive-cost line
    first_drive_cost_index = None

    # Flag to indicate parsing inside the objects section
    in_objects_section = False

    for i, line in enumerate(lines):
        if '(define (problem ' in line:
            lines[i] = f'(define (problem {base_file_name})\n'
        elif '(:objects' in line:
            in_objects_section = True
        elif in_objects_section:
            if '-' in line:
                object_type = line.split('-')[-1].strip()
                if object_type in ['market', 'depot']:
                    objects = line.split('-')[0].split()
                    markets_depots.update(objects)
            if ')' in line:  # End of objects section
                in_objects_section = False
        elif '(= (on-sale goods' in line and ' 0)' in line:
            parts = line.split()
            goods_market = parts[2], parts[3].rstrip(')')
            goods_not_on_sale[goods_market] = (i, indentation)
        elif '(= (price goods' in line:
            parts = line.split()
            goods_market = parts[2], parts[3].rstrip(')')
            goods_with_price.add(goods_market)
        elif '(= (drive-cost ' in line:
            parts = line.split()
            from_to = parts[2], parts[3].rstrip(')')
            if from_to[0] == from_to[1]:
                existing_self_drive_costs.add(from_to[0])
            if first_drive_cost_index is None:
                first_drive_cost_index = i

    # Create a list of new price lines to insert
    new_price_lines = []

    for goods_market, (line_index, indentation) in goods_not_on_sale.items():
        if goods_market not in goods_with_price:
            new_price_line = '\t' * indentation + f'(= (price {goods_market[0]} {goods_market[1]}) 0)\n'
            new_price_lines.append((line_index, new_price_line))

    # Sort new price lines in reverse order of their line numbers
    new_price_lines.sort(reverse=True)

    # Number of lines added
    lines_added = 0

    # Insert new price lines into the file content
    for line_index, new_price_line in new_price_lines:
        lines.insert(line_index + lines_added, new_price_line)
        lines_added += 1

    # Update the first drive-cost index after adding new price lines
    first_drive_cost_index += lines_added

    # Insert self drive-cost lines if missing
    for loc in markets_depots:
        if loc not in existing_self_drive_costs:
            drive_cost_line = '\t(= (drive-cost ' + loc + ' ' + loc + ') 0)\n'
            lines.insert(first_drive_cost_index, drive_cost_line)
            first_drive_cost_index += 1

    return ''.join(lines)

def main():
    parser = argparse.ArgumentParser(description='Adjust prices in the file for goods not on sale, update problem name, and ensure self drive-costs.')
    parser.add_argument('file', help='Path to the file to be processed')
    args = parser.parse_args()

    fixed = adjust_prices(args.file)
    with open(args.file, 'w') as file:
        file.write(fixed)

if __name__ == "__main__":
    main()
