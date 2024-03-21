import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import essentia
import essentia.standard as es

eps = np.finfo(float).eps

def extract_features(audio_file, sample_rate=48000, start=0, stop=-1):
    """ Given a path to an audio file extract the relevant features.

    Args:
        audio_file (str): Path to valid audio file.
        sample_rate (int): Desired analysis sample rate of the audio.
        start (int): Startinig point to load from the file in seconds. 
        stop (int): End point to stop loading audio from the file in seconds. 
            -1 denotes load til the end of the file

    Returns:
        features (dict): Containing the 33 extracted features.

    Feature list:
        - Crest factor 100ms and 1s
        - Sub-band flux
        - EBU R128 Loudness
        - PMF (temporal) Kurtosis
        - PMF (temporal) Skew
        - PMF (temporal) Centroid
        - PMF (temporal) Spread
        - Spectral Kurtosis
        - Spectral Skew
        - Spectral Centroid
        - Spectral Spread
        - Spectral Entropy
        - Spectral Rolloff (0.85 and 0.95)
        - Stereo Panning (0-3)
        - L/R Balance
        - Side-Mid ratio
    """

    loaderstereo = es.AudioLoader(filename=audio_file)
    loadermono = es.EasyLoader(downmix='mix', filename=audio_file, sampleRate=sample_rate)

    # construct all the algorithms needed for extraction
    crest = es.Crest()
    temporalPMF = es.Histogram(maxValue=2, minValue=0, numberBins=201, normalize="unit_sum")
    cmoments = es.CentralMoments()
    dshape = es.DistributionShape()
    centroid = es.Centroid()
    entropy = es.Entropy()
    power = es.InstantPower()
    melbands = es.MelBands()
    flux = es.Flux()
    rolloff85 = es.RollOff(cutoff=0.85, sampleRate=sample_rate)
    rolloff95 = es.RollOff(cutoff=0.95, sampleRate=sample_rate)
    panning = es.Panning(sampleRate=sample_rate, numCoeffs=4)
    demuxer = es.StereoDemuxer()
    loudness = es.LoudnessEBUR128(sampleRate=sample_rate)
    windowing = es.Windowing(type='blackmanharris62')
    spectrum = es.Spectrum()
    poolagg = es.PoolAggregator(defaultStats=["mean"])
    
    # create 10 band filterbank 
    band0 = es.BandPass(bandwidth=50,  cutoffFrequency=25, sampleRate=sample_rate)
    band1 = es.BandPass(bandwidth=50,  cutoffFrequency=75, sampleRate=sample_rate)
    band2 = es.BandPass(bandwidth=100, cutoffFrequency=150, sampleRate=sample_rate)
    band3 = es.BandPass(bandwidth=200, cutoffFrequency=300, sampleRate=sample_rate)
    band4 = es.BandPass(bandwidth=400, cutoffFrequency=600, sampleRate=sample_rate)
    band5 = es.BandPass(bandwidth=800, cutoffFrequency=1200, sampleRate=sample_rate)
    band6 = es.BandPass(bandwidth=1600, cutoffFrequency=2400, sampleRate=sample_rate)
    band7 = es.BandPass(bandwidth=3200, cutoffFrequency=4800, sampleRate=sample_rate)
    band8 = es.BandPass(bandwidth=6400, cutoffFrequency=9600, sampleRate=sample_rate)
    band9 = es.BandPass(bandwidth=12800, cutoffFrequency=19200, sampleRate=sample_rate)
    
    # load the mono audio file
    mono_audio = loadermono()
    stereo_audio, sample_rate, num_ch, md5, bit_rate, codec = loaderstereo()

    # index the proper portion of the audio
    startsamp = int(start * sample_rate)
    stopsamp  = int(stop * sample_rate)

    if stop == -1:
        mono_audio = mono_audio[startsamp:]
        stereo_audio = stereo_audio[startsamp:,:]
    else:
        mono_audio = mono_audio[startsamp:stopsamp]
        stereo_audio = stereo_audio[startsamp:stopsamp,:]

    # pool to store extracted feature frames
    pool = essentia.Pool()

    # dict to store the final feature vector
    features = {}

    # calculate loudness statistics over the whole audio file
    m, s, i, r = loudness(stereo_audio)
    features["momentaryLoudness.mean"] = np.mean(m)
    features["shortTermLoudness.mean"] = np.mean(s)
    features["integratedLoudness"] = i
    features["loudnessRange"] = r

    # calculate the temporal PMF for the whole audio file
    pmf, binedges = temporalPMF(mono_audio + 1)
    pmfmoments = cmoments(pmf)
    pmfcentroid = centroid(pmf)
    pmfspread, pmfskewness, pmfkurtosis = dshape(pmfmoments)
    features["pmfSpread"] = pmfspread
    features["pmfSkewness"] = pmfskewness
    features["pmfKurtosis"] = pmfkurtosis
    features["pmfCentroid"] = pmfcentroid

    # first we use window of about 86 ms
    for frame in es.FrameGenerator(mono_audio, frameSize=4096, hopSize=2048, startFromZero=True):
        crest100ms = crest(np.abs(frame))
        frameSpectrum = spectrum(windowing(frame)) + eps
        dBframeSpectrum = 20 * np.log10(frameSpectrum)

        spectrumMoments = cmoments(frameSpectrum)
        spectralCentroid = centroid(frameSpectrum)
        spectralEntropy = entropy(frameSpectrum)
        spectralRollOff85 = rolloff85(frameSpectrum)
        spectralRollOff95 = rolloff95(frameSpectrum)
        spectralSpread, spectralSkewness, spectralKurtosis = dshape(spectrumMoments)

        band0spectrum = spectrum(windowing(band0(frame))) + eps
        band1spectrum = spectrum(windowing(band1(frame))) + eps
        band2spectrum = spectrum(windowing(band2(frame))) + eps
        band3spectrum = spectrum(windowing(band3(frame))) + eps
        band4spectrum = spectrum(windowing(band4(frame))) + eps
        band5spectrum = spectrum(windowing(band5(frame))) + eps
        band6spectrum = spectrum(windowing(band6(frame))) + eps
        band7spectrum = spectrum(windowing(band7(frame))) + eps
        band8spectrum = spectrum(windowing(band8(frame))) + eps
        band9spectrum = spectrum(windowing(band9(frame))) + eps

        band0flux = flux(band0spectrum)
        band1flux = flux(band1spectrum)
        band2flux = flux(band2spectrum)
        band3flux = flux(band3spectrum)
        band4flux = flux(band4spectrum)
        band5flux = flux(band5spectrum)
        band6flux = flux(band6spectrum)
        band7flux = flux(band7spectrum)
        band8flux = flux(band8spectrum)
        band9flux = flux(band9spectrum)

        # pool feature values over the frames
        pool.add("crest100ms", crest100ms)
        pool.add("spectralCentroid", spectralCentroid)
        pool.add("spectralEntropy", spectralEntropy)
        pool.add("spectralSpread", spectralSpread)
        pool.add("spectralSkewness", spectralSkewness)
        pool.add("spectralKurtosis", spectralKurtosis)
        pool.add("spectralRollOff85", spectralRollOff85)
        pool.add("spectralRollOff95", spectralRollOff95)
        pool.add("subBandFlux0", band0flux)
        pool.add("subBandFlux1", band1flux)
        pool.add("subBandFlux2", band2flux)
        pool.add("subBandFlux3", band3flux)
        pool.add("subBandFlux4", band4flux)
        pool.add("subBandFlux5", band5flux)
        pool.add("subBandFlux6", band6flux)
        pool.add("subBandFlux7", band7flux)
        pool.add("subBandFlux8", band8flux)
        pool.add("subBandFlux9", band9flux)

    # Now we need to analyse the stereo singal for some features
    audio_left, audio_right = demuxer(stereo_audio)
    for frameLeft, frameRight in zip(es.FrameGenerator(audio_left, frameSize=4096, hopSize=2048, startFromZero=True),
                                     es.FrameGenerator(audio_right, frameSize=4096, hopSize=2048, startFromZero=True)):    

        frameLeftSpectrum = spectrum(windowing(frameLeft)) + eps
        frameRightSpectrum = spectrum(windowing(frameRight)) + eps
        panningCoeffs = panning(frameLeftSpectrum, frameRightSpectrum)

        # pool feature values over the frames
        pool.add("panningCoeffs0", panningCoeffs[0][0])
        pool.add("panningCoeffs1", panningCoeffs[0][1])
        pool.add("panningCoeffs2", panningCoeffs[0][2])
        pool.add("panningCoeffs3", panningCoeffs[0][3])

    # now we use window of about 1 sec
    for frame in es.FrameGenerator(mono_audio, frameSize=int(sample_rate), hopSize=int(sample_rate//2), startFromZero=True):
        crest1sec = crest(np.abs(frame))
        pool.add("crest1sec", crest1sec)

    # compute the left/right imbalance
    leftPower = power(audio_left)
    rightPower = power(audio_right)
    lrimbalance = (rightPower - leftPower) / (rightPower + leftPower)
    features["LRImbalance"] = lrimbalance

    # compute the mid/side ratio
    sidePower = power((audio_left + audio_right)/2)
    midPower = power((audio_left - audio_right)/2)
    if midPower == 0:
        features["midSideRatio"] = np.nan
    else:
        midSideRatio = sidePower / midPower
        features["midSideRatio"] = midSideRatio

    # aggregate the pool and add final features 
    avgpool = poolagg(pool)
    for descriptor in avgpool.descriptorNames():
        features[descriptor] = avgpool[descriptor]

    return features

def extract_audio_features(audio_files, feature_type="music-extractor"):
    """ Given a list of paths of audio files, extract a set of features with essentia.

    Args:
        audio_files (list): List of strings pointing to audio files.
        feature_type (str): Either 'music-extractor' or 'med'.

        'music-extractor' - extracts all of the standard features.
        'med' - extracts the features defined for analysis of Mix Evaluation Dataset. 

    """

    print(len(audio_files), "found in total")

    dataset = {}
    processed = []
    
    if not os.path.isfile("/homes/ssv02/mix_engineer_style/data/processed_eras.txt"):
        print("Creating processed.txt file...")
        #if file doesn exist create one
        processed = []
    else:
        with open("/homes/ssv02/mix_engineer_style/data/processed_eras.txt", "r") as fp:
            print("Reading processed.txt file...")
            processed = [tracker.strip('\n') for tracker in fp.readlines()]
            print(processed)

    for n, audio_file in enumerate(audio_files):
        print(audio_file)
        

        sys.stdout.write(f"Extracted features from {n}/{len(audio_files)}...\r")
        sys.stdout.flush()
        print()

        # get the songname and engineer name
        songname = os.path.basename(audio_file).strip('.wav')
        eng_name = audio_file.split('/')[-2]
        

        # create a unique identifier for the song
        tracker = f"{songname}-{eng_name}"
        
        #check if the song is already processed
        if tracker in processed:
            print(f"\n {tracker} already processed")
            continue
        else:
            print(tracker)

        # extract the features
        if feature_type == "music-extractor":
            features, features_frames = es.MusicExtractor(lowlevelFrameSize = 2048,
                                                            lowlevelHopSize = 1024,
                                                            lowlevelStats = ['mean', 'stdev'])(audio_file)
        elif feature_type == "med":
            
            sample_data = {'era_name' : eng_name,
                            'songname' : songname}
    
            features = extract_features(audio_file)
            for feature_name, feature in features.items():
                sample_data[feature_name] = features[feature_name]   
            
            dataset[n] = sample_data
            
            
        else:
            raise ValueError(f"Invalid 'feature_type' {feature_type}.")
        
        processed.append(tracker)
        
        with open('/homes/ssv02/mix_engineer_style/data/processed_eras.txt', 'a') as fp:
            fp.write(tracker)
            fp.write("\n")
            print(f"\n added {tracker} to processed_eras.txt")
            fp.close()

    outputfile = os.path.join('/homes/ssv02/mix_engineer_style/data/features_eras.csv')

    if not os.path.isfile(outputfile):
        df = pd.DataFrame(dataset)
        df = df.transpose()
        df.to_csv(outputfile)
        print(f"\n added to csv")
       


def prune_features(df):

    # remove unwanted coloumns
    with open('data/feature_columns.txt', 'r') as file:
        feature_columns = file.read().split('\n')

    df = df[feature_columns]

    return df

def find_segment_idx():
    #need to find a segmentation code that can find chorus and verse index but right now no suport for that. Removed the code from extract_audio_features as well. 
    return


if __name__ == '__main__':
    #audio engineers
    #audio_files = glob.glob("/import/c4dm-datasets-ext/mix_eng_songs/playlists_wav/**/*.wav")

    #eras 
    audio_files= glob.glob("/import/c4dm-datasets-ext/mix_eng_songs/era_based_songs_wav/**/*.wav")
    print(f"\nFound a total of {len(audio_files)} audio files...")
    

    extract_audio_features(audio_files, feature_type='med')
    # feature_df = pd.read_csv('/homes/ssv02/mix_engineer_style/data/med_mix_features.csv')
    # feature_df = prune_features(feature_df)

    # eval_df = pd.read_csv("data/med/mix_eval_dataset.csv")

    # feature_df.to_csv("data/med/med_mix_features_with_scores.csv")