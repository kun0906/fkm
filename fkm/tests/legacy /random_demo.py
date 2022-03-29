# import numpy as np
#
# for i in range(5):
#     print()
#     np.random.seed(i)
#     for iteration in range(1, 1 + 10):
#         # clients_in_round = random.sample(x, clients_per_round) # without replacement and random
#         # r=np.random.RandomState(iteration)
#         r = np.random.RandomState((i+1)*iteration)
#         clients_in_round = r.choice(range(1, 100), size=3, replace=False)
#         print(iteration, clients_in_round)
#     # print(np.random.get_state())


def dc(nums):
    if len(nums) < 2:
        return (0, nums)

    m = len(nums) // 2
    left, nums1 = dc(nums[:m])
    right, nums2 = dc(nums[m:])

    res = 0
    # for i in range(len(nums1)):
    #     for j in range(len(nums2)):
    #         if nums1[i] > 2 * nums2[j]:
    #             res +=1
    # nums = sorted(nums)
    print(nums1)
    # nums1 = merge_sort(nums1, nums2)

    return (left + right + res, nums1 + nums2)

nums = [1,3,2,3,1]
cnt, _ = dc(nums)





