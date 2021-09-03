from pydub import AudioSegment
import random
import os
import time


global_check_set = set()


def merge_audio(path, speakers):
    speakers.sort()

    audio_str = ''
    audio_list = []
    for i in speakers:
        audio_path = random.sample(os.listdir(path + '/' + i), 1)
        single_audio = path + '/' + i + '/' + audio_path[0]
        audio_list.append(single_audio)
        audio_str += audio_path[0] + ' '

    while audio_str in global_check_set:
        print("Duplicate Audio Combination: ", audio_str)
        audio_str = ''
        audio_list = []
        for i in speakers:
            audio_path = random.sample(os.listdir(path + '/' + i), 1)
            single_audio = path + '/' + i + '/' + audio_path[0]
            audio_list.append(single_audio)
            audio_str += audio_path[0] + ' '

    global_check_set.add(audio_str)
    print("Final Audio Combination: ", audio_str)
    combined = 0
    for i in audio_list:
        combined += AudioSegment.from_file(i, format="flac")
    return combined


def populate_merged_audio(path, n):
    print("Home Path: ", path)
    speakers = os.listdir(path + '/speakers')
    print("Speakers: ", speakers)
    print('Total # of speakers: ', len(speakers))
    start_time = time.time()

    for i in range(1, 40001):
        print("#: ", i)
        print("========")
        combination_of_speaker = random.sample(speakers, n)
        print("Randomized Combination: ", combination_of_speaker)
        combined_audio = merge_audio(path + '/speakers', combination_of_speaker)
        combined_audio.export(path + '/exported/' + str(n) + 'speakers/' + str(i) + '.wav', format="wav")
        print("\n")

    print("--- %s seconds ---" % (time.time() - start_time))


def handle_one_speaker_case(path, n):
    print("Home Path: ", path)
    speakers = os.listdir(path + '/speakers')
    print("Speakers: ", speakers)
    print('Total # of speakers: ', len(speakers))
    start_time = time.time()

    count = 1
    for i in speakers:
        files = os.listdir(path + '/speakers' + '/' + i)
        for j in files:
            print("#: ", count)
            print("========")
            print("Final Audio Combination: ", j)
            converted_audio = AudioSegment.from_file(path + '/speakers' + '/' + i + '/' + j, format="flac")
            converted_audio.export(path + '/exported/' + str(n) + 'speakers/' + str(count) + '.wav', format="wav")
            count += 1
            print("\n")
    print("--- %s seconds ---" % (time.time() - start_time))


def get_combined_audio(path, speakers, audio=None):
    audio_path_list_speaker_1 = random.sample(os.listdir(path + '/' + speakers[0]), 3)
    audio_path_list_speaker_2 = random.sample(os.listdir(path + '/' + speakers[1]), 3)

    print("Speaker 1 Audio List: ", audio_path_list_speaker_1)
    print("Speaker 2 Audio List: ", audio_path_list_speaker_2)

    audio = AudioSegment.empty()
    audio += AudioSegment.from_file(path + '/' + speakers[0] + '/' + audio_path_list_speaker_1[0], format="flac")
    audio += AudioSegment.from_file(path + '/' + speakers[1] + '/' + audio_path_list_speaker_2[0], format="flac")
    audio += AudioSegment.from_file(path + '/' + speakers[0] + '/' + audio_path_list_speaker_1[1], format="flac")
    audio += AudioSegment.from_file(path + '/' + speakers[1] + '/' + audio_path_list_speaker_2[1], format="flac")
    audio += AudioSegment.from_file(path + '/' + speakers[0] + '/' + audio_path_list_speaker_1[2], format="flac")
    audio += AudioSegment.from_file(path + '/' + speakers[1] + '/' + audio_path_list_speaker_2[2], format="flac")

    return audio


def generate_two_speaker_audio_data(path):
    print("Home Path: ", path)
    speakers = os.listdir(path + '/speakers')
    print("Speakers: ", speakers)
    print('Total # of speakers: ', len(speakers))
    start_time = time.time()

    for i in range(1, 25001):
        print("#: ", i)
        print("========")

        combination_of_speaker = random.sample(speakers, 2)
        path_str = ''
        combination_of_speaker.sort()
        path_str += combination_of_speaker[0] + ' ' + combination_of_speaker[1]

        while path_str in global_check_set:
            print("Duplicate Combination: ", combination_of_speaker)
            combination_of_speaker = random.sample(speakers, 2)
            path_str = ''
            combination_of_speaker.sort()
            path_str += combination_of_speaker[0] + ' ' + combination_of_speaker[1]

        print("Final Combination: ", combination_of_speaker)
        global_check_set.add(path_str)
        combined_audio = get_combined_audio(path + '/speakers', combination_of_speaker)
        combined_audio.export(path + '/exported/v2/2speakers/' + str(i) + '.wav', format="wav")
        print("\n")

    print("--- %s seconds ---" % (time.time() - start_time))


def get_combined_audio_3(path, speakers, audio=None):
    audio_path_list_speaker_1 = random.sample(os.listdir(path + '/' + speakers[0]), 2)
    audio_path_list_speaker_2 = random.sample(os.listdir(path + '/' + speakers[1]), 2)
    audio_path_list_speaker_3 = random.sample(os.listdir(path + '/' + speakers[2]), 2)

    print("Speaker 1 Audio List: ", audio_path_list_speaker_1)
    print("Speaker 2 Audio List: ", audio_path_list_speaker_2)
    print("Speaker 3 Audio List: ", audio_path_list_speaker_3)

    audio = AudioSegment.empty()
    audio += AudioSegment.from_file(path + '/' + speakers[0] + '/' + audio_path_list_speaker_1[0], format="flac")
    audio += AudioSegment.from_file(path + '/' + speakers[1] + '/' + audio_path_list_speaker_2[0], format="flac")
    audio += AudioSegment.from_file(path + '/' + speakers[2] + '/' + audio_path_list_speaker_3[0], format="flac")
    audio += AudioSegment.from_file(path + '/' + speakers[0] + '/' + audio_path_list_speaker_1[1], format="flac")
    audio += AudioSegment.from_file(path + '/' + speakers[2] + '/' + audio_path_list_speaker_3[1], format="flac")
    audio += AudioSegment.from_file(path + '/' + speakers[1] + '/' + audio_path_list_speaker_2[1], format="flac")

    return audio


def generate_three_speaker_audio_data(path):
    print("Home Path: ", path)
    speakers = os.listdir(path + '/speakers')
    print("Speakers: ", speakers)
    print('Total # of speakers: ', len(speakers))
    start_time = time.time()

    for i in range(1, 25001):
        print("#: ", i)
        print("========")

        combination_of_speaker = random.sample(speakers, 3)
        path_str = ''
        combination_of_speaker.sort()
        path_str += combination_of_speaker[0] + ' ' + combination_of_speaker[1] + ' ' + combination_of_speaker[2]

        while path_str in global_check_set:
            print("Duplicate Combination: ", combination_of_speaker)
            combination_of_speaker = random.sample(speakers, 3)
            path_str = ''
            combination_of_speaker.sort()
            path_str += combination_of_speaker[0] + ' ' + combination_of_speaker[1] + ' ' + combination_of_speaker[2]

        print("Final Combination: ", combination_of_speaker)
        global_check_set.add(path_str)
        combined_audio = get_combined_audio_3(path + '/speakers', combination_of_speaker)
        combined_audio.export(path + '/exported/v2/3speakers/' + str(i) + '.wav', format="wav")
        print("\n")

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    #populate_merged_audio("/home/shuvornb/Desktop/NIJ-AI-SMS/testdata", 1)
    #populate_merged_audio("/home/shuvornb/Desktop/NIJ-AI-SMS/testdata", 2)
    #populate_merged_audio("/home/shuvornb/Desktop/NIJ-AI-SMS/testdata", 3)
    #print("Global Set Size: ", len(global_check_set))

    #handle_one_speaker_case("/home/shuvornb/Desktop/NIJ-AI-SMS/testdata", 1)

    #generate_two_speaker_audio_data("/home/shuvornb/Desktop/NIJ-AI-SMS/testdata")
    generate_three_speaker_audio_data("/home/shuvornb/Desktop/NIJ-AI-SMS/testdata")


