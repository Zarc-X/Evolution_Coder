#!/usr/bin/env python3
"""
创建示例种子数据
用于快速测试简化版 Tree-of-Evolution
"""

import json
import os

# 示例种子数据：50个简单的Python代码片段
SEED_SAMPLES = [
    {"id": "1", "content": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"},
    {"id": "2", "content": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"},
    {"id": "3", "content": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"},
    {"id": "4", "content": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"},
    {"id": "5", "content": "def reverse_string(s):\n    return s[::-1]"},
    {"id": "6", "content": "def count_words(text):\n    return len(text.split())"},
    {"id": "7", "content": "def find_max(arr):\n    return max(arr) if arr else None"},
    {"id": "8", "content": "def is_palindrome(s):\n    return s == s[::-1]"},
    {"id": "9", "content": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a"},
    {"id": "10", "content": "def lcm(a, b):\n    return abs(a * b) // gcd(a, b)"},
    {"id": "11", "content": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr"},
    {"id": "12", "content": "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)"},
    {"id": "13", "content": "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)"},
    {"id": "14", "content": "def power(base, exponent):\n    result = 1\n    for _ in range(exponent):\n        result *= base\n    return result"},
    {"id": "15", "content": "def sum_digits(n):\n    return sum(int(digit) for digit in str(n))"},
    {"id": "16", "content": "def remove_duplicates(arr):\n    return list(set(arr))"},
    {"id": "17", "content": "def flatten_list(nested_list):\n    result = []\n    for item in nested_list:\n        if isinstance(item, list):\n            result.extend(flatten_list(item))\n        else:\n            result.append(item)\n    return result"},
    {"id": "18", "content": "def count_characters(text):\n    return {char: text.count(char) for char in set(text)}"},
    {"id": "19", "content": "def is_anagram(s1, s2):\n    return sorted(s1) == sorted(s2)"},
    {"id": "20", "content": "def capitalize_words(text):\n    return ' '.join(word.capitalize() for word in text.split())"},
    {"id": "21", "content": "def find_common_elements(arr1, arr2):\n    return list(set(arr1) & set(arr2))"},
    {"id": "22", "content": "def rotate_array(arr, k):\n    k = k % len(arr)\n    return arr[-k:] + arr[:-k]"},
    {"id": "23", "content": "def transpose_matrix(matrix):\n    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]"},
    {"id": "24", "content": "def matrix_multiply(A, B):\n    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]"},
    {"id": "25", "content": "def depth_first_search(graph, start):\n    visited, stack = set(), [start]\n    while stack:\n        vertex = stack.pop()\n        if vertex not in visited:\n            visited.add(vertex)\n            stack.extend(graph[vertex] - visited)\n    return visited"},
    {"id": "26", "content": "def breadth_first_search(graph, start):\n    visited, queue = set(), [start]\n    while queue:\n        vertex = queue.pop(0)\n        if vertex not in visited:\n            visited.add(vertex)\n            queue.extend(graph[vertex] - visited)\n    return visited"},
    {"id": "27", "content": "def dijkstra(graph, start):\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0\n    unvisited = set(graph.keys())\n    while unvisited:\n        current = min(unvisited, key=lambda node: distances[node])\n        unvisited.remove(current)\n        for neighbor, weight in graph[current].items():\n            if neighbor in unvisited:\n                distances[neighbor] = min(distances[neighbor], distances[current] + weight)\n    return distances"},
    {"id": "28", "content": "def validate_email(email):\n    import re\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return re.match(pattern, email) is not None"},
    {"id": "29", "content": "def validate_phone(phone):\n    import re\n    pattern = r'^\\+?1?\\d{9,15}$'\n    return re.match(pattern, phone) is not None"},
    {"id": "30", "content": "def extract_numbers(text):\n    import re\n    return [int(x) for x in re.findall(r'\\d+', text)]"},
    {"id": "31", "content": "def word_frequency(text):\n    words = text.lower().split()\n    return {word: words.count(word) for word in set(words)}"},
    {"id": "32", "content": "def longest_common_subsequence(s1, s2):\n    m, n = len(s1), len(s2)\n    dp = [[0] * (n + 1) for _ in range(m + 1)]\n    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n            if s1[i-1] == s2[j-1]:\n                dp[i][j] = dp[i-1][j-1] + 1\n            else:\n                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n    return dp[m][n]"},
    {"id": "33", "content": "def knapsack(weights, values, capacity):\n    n = len(weights)\n    dp = [[0] * (capacity + 1) for _ in range(n + 1)]\n    for i in range(1, n + 1):\n        for w in range(1, capacity + 1):\n            if weights[i-1] <= w:\n                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])\n            else:\n                dp[i][w] = dp[i-1][w]\n    return dp[n][capacity]"},
    {"id": "34", "content": "def edit_distance(s1, s2):\n    m, n = len(s1), len(s2)\n    dp = [[0] * (n + 1) for _ in range(m + 1)]\n    for i in range(m + 1):\n        for j in range(n + 1):\n            if i == 0:\n                dp[i][j] = j\n            elif j == 0:\n                dp[i][j] = i\n            elif s1[i-1] == s2[j-1]:\n                dp[i][j] = dp[i-1][j-1]\n            else:\n                dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])\n    return dp[m][n]"},
    {"id": "35", "content": "def coin_change(coins, amount):\n    dp = [float('inf')] * (amount + 1)\n    dp[0] = 0\n    for coin in coins:\n        for i in range(coin, amount + 1):\n            dp[i] = min(dp[i], dp[i - coin] + 1)\n    return dp[amount] if dp[amount] != float('inf') else -1"},
    {"id": "36", "content": "def longest_increasing_subsequence(arr):\n    if not arr:\n        return 0\n    dp = [1] * len(arr)\n    for i in range(1, len(arr)):\n        for j in range(i):\n            if arr[j] < arr[i]:\n                dp[i] = max(dp[i], dp[j] + 1)\n    return max(dp)"},
    {"id": "37", "content": "def two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return [seen[complement], i]\n        seen[num] = i\n    return []"},
    {"id": "38", "content": "def three_sum(nums):\n    nums.sort()\n    result = []\n    for i in range(len(nums) - 2):\n        if i > 0 and nums[i] == nums[i-1]:\n            continue\n        left, right = i + 1, len(nums) - 1\n        while left < right:\n            s = nums[i] + nums[left] + nums[right]\n            if s < 0:\n                left += 1\n            elif s > 0:\n                right -= 1\n            else:\n                result.append([nums[i], nums[left], nums[right]])\n                while left < right and nums[left] == nums[left+1]:\n                    left += 1\n                while left < right and nums[right] == nums[right-1]:\n                    right -= 1\n                left += 1\n                right -= 1\n    return result"},
    {"id": "39", "content": "def max_subarray_sum(arr):\n    max_sum = current_sum = arr[0]\n    for num in arr[1:]:\n        current_sum = max(num, current_sum + num)\n        max_sum = max(max_sum, current_sum)\n    return max_sum"},
    {"id": "40", "content": "def product_except_self(nums):\n    n = len(nums)\n    result = [1] * n\n    for i in range(1, n):\n        result[i] = result[i-1] * nums[i-1]\n    right = 1\n    for i in range(n-1, -1, -1):\n        result[i] *= right\n        right *= nums[i]\n    return result"},
    {"id": "41", "content": "def group_anagrams(strs):\n    from collections import defaultdict\n    groups = defaultdict(list)\n    for s in strs:\n        key = ''.join(sorted(s))\n        groups[key].append(s)\n    return list(groups.values())"},
    {"id": "42", "content": "def top_k_frequent(nums, k):\n    from collections import Counter\n    return [num for num, _ in Counter(nums).most_common(k)]"},
    {"id": "43", "content": "def merge_intervals(intervals):\n    if not intervals:\n        return []\n    intervals.sort(key=lambda x: x[0])\n    merged = [intervals[0]]\n    for current in intervals[1:]:\n        if current[0] <= merged[-1][1]:\n            merged[-1][1] = max(merged[-1][1], current[1])\n        else:\n            merged.append(current)\n    return merged"},
    {"id": "44", "content": "def meeting_rooms(intervals):\n    intervals.sort(key=lambda x: x[0])\n    for i in range(1, len(intervals)):\n        if intervals[i][0] < intervals[i-1][1]:\n            return False\n    return True"},
    {"id": "45", "content": "def reverse_linked_list(head):\n    prev = None\n    current = head\n    while current:\n        next_node = current.next\n        current.next = prev\n        prev = current\n        current = next_node\n    return prev"},
    {"id": "46", "content": "def has_cycle(head):\n    slow = fast = head\n    while fast and fast.next:\n        slow = slow.next\n        fast = fast.next.next\n        if slow == fast:\n            return True\n    return False"},
    {"id": "47", "content": "def merge_two_sorted_lists(l1, l2):\n    dummy = ListNode(0)\n    current = dummy\n    while l1 and l2:\n        if l1.val <= l2.val:\n            current.next = l1\n            l1 = l1.next\n        else:\n            current.next = l2\n            l2 = l2.next\n        current = current.next\n    current.next = l1 or l2\n    return dummy.next"},
    {"id": "48", "content": "def invert_binary_tree(root):\n    if not root:\n        return None\n    root.left, root.right = invert_binary_tree(root.right), invert_binary_tree(root.left)\n    return root"},
    {"id": "49", "content": "def max_depth_binary_tree(root):\n    if not root:\n        return 0\n    return 1 + max(max_depth_binary_tree(root.left), max_depth_binary_tree(root.right))"},
    {"id": "50", "content": "def is_valid_bst(root, min_val=float('-inf'), max_val=float('inf')):\n    if not root:\n        return True\n    if root.val <= min_val or root.val >= max_val:\n        return False\n    return is_valid_bst(root.left, min_val, root.val) and is_valid_bst(root.right, root.val, max_val)"},
]


def main():
    """创建种子数据文件"""
    os.makedirs("data/seed", exist_ok=True)
    output_file = "data/seed/seed_samples.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(SEED_SAMPLES, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 已创建种子数据文件: {output_file}")
    print(f"✓ 包含 {len(SEED_SAMPLES)} 个样本")
    print("\n现在可以运行: python run_pipeline.py")


if __name__ == "__main__":
    main()

