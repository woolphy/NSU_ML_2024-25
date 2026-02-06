def binary_search(arr, key):
    l = len(arr)
    if l == 0:
        return None
    hight = l - 1
    low = 0
    middle = l // 2
    t = 0
    while (key != arr[middle]) and (low <= hight):
        if key > arr[middle]:
            low = middle + 1
        else:
            hight = middle - 1
        middle = (hight + low) // 2
    if key == arr[middle]:
            for i in range(1, middle):
                if arr[middle - i] == arr[middle]:
                    t = middle - i
    if low > hight:
        return None
    if t != 0:
        return t
    else:
        return middle


assert binary_search([1, 4, 5], 1) == 0
assert binary_search([1, 4, 5, 10, 20], 20) == 4
assert binary_search([1, 5, 8, 10], 10) == 3
assert binary_search([3, 5, 9, 10], 9) == 2
assert binary_search([], 5) == None
assert binary_search([1, 3, 3, 3, 6, 8], 3) == 1