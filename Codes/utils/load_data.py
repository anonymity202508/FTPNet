import json
import datetime
import math


def load_data(data_paths):
    labels = []
    posts_by_user = []

    # 删除非原创的微博，仅保留用户标签、微博发布的内容和时间
    for data_path in data_paths:
        with open(data_path, "r", encoding='utf8') as f:
            reader = json.load(f)
            for user in reader:
                posts = []
                for post in user['tweets']:
                    if post['tweet_is_original'] == 'True' and len(post['tweet_content']) >= 2:
                        # isinstance(post['tweet_content'], 'NoneType')
                        if len(post['tweet_content']) <= 128:
                            posts.append({'tweet_content': post['tweet_content'], 'posting_time': datetime.datetime.strptime(post['posting_time'][0:16], '%Y-%m-%d %H:%M')})
                        else:
                            iter_post = math.ceil(len(post['tweet_content'])/128)
                            for i in range(iter_post - 1):
                                posts.append({'tweet_content': post['tweet_content'][i * 128:i*128 + 128],
                                              'posting_time': datetime.datetime.strptime(post['posting_time'][0:16],
                                                                                         '%Y-%m-%d %H:%M')})
                            posts.append({'tweet_content': post['tweet_content'][iter_post * 128:],
                                          'posting_time': datetime.datetime.strptime(post['posting_time'][0:16],
                                                                                     '%Y-%m-%d %H:%M')})
                        # posts.append({'tweet_content': post['tweet_content'], 'posting_time': datetime.datetime.strptime(
                        #                   post['posting_time'][0:16], '%Y-%m-%d %H:%M')})
                if len(posts) > 10:  # 预处理，删除帖子数少于10的用户
                    posts_by_user.append(posts)
                    if 'normal' in data_path:
                        labels.append(int(0))
                    elif 'depressed' in data_path:
                        labels.append(int(1))  # label要为int型，不能是str，需要转换
    return posts_by_user, labels
