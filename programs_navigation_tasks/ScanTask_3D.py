from __future__ import absolute_import, division, print_function
from exp_functions import initial_exp, make_distribute, show_stimuli, mkImg, mkText, run_introduction, findShortest, \
    checkChoice, initialise_D3, follow_D3
from numpy.random import random, randint, normal, shuffle, choice as randchoice
from psychopy import visual, event, core, gui, data, logging, clock
from psychopy.hardware import keyboard
import os, time

# prepare materials
expName = 'scan_3Dnavigation'
sessions = ['hat', 'dog']
thisExp, expInfo, win, defaultKeyboard, filename, endExpNow = initial_exp(expName, sessions, 'cm', [1280, 720],
                                                                          fullscreen=False)

nRoute = 10
space_shape = [6, 6, 6]
ISI1 = 3.0  # from current to goal
ISI2 = 4.0  # from goal to options
ITI = [3, 4, 5, 6, 7, 8, 9]  # interval between trials
IPI = [6, 7, 8, 9, 10, 11, 12, 13]  # interval between paths
time_feedback = 2.0
prob_break = 0.2
prob_block = 0.0
session = expInfo['session']
material = rf'materials/{session}List.xlsx'
all_dim_list = ['D1', 'D2', 'D3', 'D4']
dim_list = randchoice(all_dim_list, 3, replace=False)
keyList = ['1', '2', '3', '4']

intro = mkImg(win, r'materials\intro_imgs\scan_D2.png')
image_coin = mkImg(win, r'materials\coin.png', (1.26, 0.8), (14, 8.78))
image_current = mkImg(win, None, (9.53, 9.53), (-8, 2.605))
image_goal = mkImg(win, None, (9.53, 9.53), (8, 2.605))
image_D = mkImg(win, None, (5.72, 5.72), (-12.04, -6.75))
image_F = mkImg(win, None, (5.72, 5.72), (-4, -6.75))
image_J = mkImg(win, None, (5.72, 5.72), (4, -6.75))
image_K = mkImg(win, None, (5.72, 5.72), (12.03, -6.75))
text_current = mkText(win, 'Current', 'Arial', (-8, 6.55), 1, 'black')
text_goal = mkText(win, 'Goal', 'Arial', (8, 6.55), 1, 'black')
text_coin = mkText(win, None, 'Arial', (16, 8.88), 0.8, 'white')
image_break = mkImg(win, r'materials\intro_imgs\break.png')
image_achieve = mkImg(win, r'materials\intro_imgs\achi.png')
image_fixation = mkImg(win, r'materials\intro_imgs\fixation.png')

# **********************exp begins***********************
gloTime = time.time()
run_introduction(win, thisExp, intro, gloTime, 's')
thisExp.addData('dimension', dim_list)
# ----------------------trials------------------------
coin = 0
trial_type = 'normal'
for thisRun in range(nRoute):
    thisIPI = randchoice(IPI)
    image_fixation.draw()
    win.flip()
    timeStart = time.time() - gloTime
    core.wait(thisIPI)
    cur, goal = initialise_D3(space_shape)
    file_list, bestChoice, finalChoices = follow_D3(material, session, trial_type, cur, goal, dim_list,
                                                                 space_shape, keyList)
    cur_file, goal_file, choiceD_file, choiceF_file, choiceJ_file, choiceK_file, bestChoice_file, corr_resp = file_list
    choiceD, choiceF, choiceJ, choiceK = finalChoices

    choice_list = ['1', '2', '3', '4']
    RouteContinue = True
    thisBreak = False
    thisExp.addData('No.Route', thisRun + 1)
    thisExp.addData('Path_start', timeStart)
    thisExp.addData('Path_interval', thisIPI)
    while RouteContinue:
        image_current.setImage(cur_file)
        image_goal.setImage(goal_file)
        image_D.setImage(choiceD_file)
        image_F.setImage(choiceF_file)
        image_J.setImage(choiceJ_file)
        image_K.setImage(choiceK_file)
        text_coin.setText(coin)
        show_stimuli([image_current, text_current, image_coin, text_coin])
        win.flip()
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        timeCurrent = time.time() - gloTime
        clock.wait(ISI1)
        show_stimuli([image_current, text_current, image_goal, text_goal, image_coin, text_coin])
        win.flip()
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        timeGoal = time.time() - gloTime
        clock.wait(ISI2)
        show_stimuli(
            [image_current, text_current, image_goal, text_goal, image_D, image_F, image_J, image_K, image_coin,
             text_coin])
        win.flip()
        trialTime = time.time()

        trial_resp = event.waitKeys(keyList=choice_list)
        rt_TimeResp = time.time() - trialTime
        timeChoice = trialTime - gloTime

        choice = trial_resp[0]

        thisExp.addData('cur_coordinate', cur)
        thisExp.addData('cur_file', cur_file)
        thisExp.addData('cur_start', timeCurrent)

        thisExp.addData('goal_coordinate', goal)
        thisExp.addData('goal_file', goal_file)
        thisExp.addData('goal_start', timeGoal)

        thisExp.addData('choiceD_file', choiceD_file)
        thisExp.addData('choiceF_file', choiceF_file)
        thisExp.addData('choiceJ_file', choiceJ_file)
        thisExp.addData('choiceK_file', choiceK_file)
        thisExp.addData('choice_start', timeChoice)

        thisExp.addData('choice', choice)
        thisExp.addData('corr_resp', corr_resp)
        thisExp.addData('rtChoice', rt_TimeResp)
        thisExp.addData('trial_type', trial_type)

        if choice == 'escape':
            win.close()
            core.quit()
        elif choice == keyList[0]:
            cur = choiceD
        elif choice == keyList[1]:
            cur = choiceF
        elif choice == keyList[2]:
            cur = choiceJ
        elif choice == keyList[3]:
            cur = choiceK

        if cur == goal:
            RouteContinue = False
            timeEndRoute = time.time() - gloTime
            thisExp.addData('RouteEnd', timeEndRoute)
        else:
            if choice != corr_resp:
                ran = make_distribute(prob_break)
                if ran == 1:
                    thisBreak = True
                    timeEndRoute = time.time() - gloTime
                    thisExp.addData('RouteEnd_time', timeEndRoute)
                    break

            block = make_distribute(prob_block)
            if block == 1:
                trial_type = 'block'
            else:
                trial_type = 'normal'
            file_list, infoList, finalChoices= follow_D3(material, session, trial_type, cur, goal,
                                                                       dim_list, space_shape, keyList)
            cur_file, goal_file, choiceD_file, choiceF_file, choiceJ_file, choiceK_file, bestChoice_file, corr_resp = file_list
            choiceD, choiceF, choiceJ, choiceK = finalChoices

        if RouteContinue == True:
            thisExp.nextEntry()

    # ----------------------achieve------------------------
    if thisBreak:
        image_break.draw()
        win.flip()
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        breakStartTime = time.time() - gloTime
        core.wait(time_feedback)
        breakEndTime = time.time() - gloTime

        thisExp.addData('breakStart', breakStartTime)
        thisExp.addData('breakEnd', breakEndTime)

    else:
        image_achieve.draw()
        win.flip()
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        timeAchi_start = time.time() - gloTime
        core.wait(time_feedback)
        timeAchi_end = time.time() - gloTime
        coin += 1

        thisExp.addData('achiStart', timeAchi_start)
        thisExp.addData('achiEnd', timeAchi_end)

    thisExp.nextEntry()

# ----------------------achieve------------------------
image_fixation.draw()
win.flip()
if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
    core.quit()
clock.wait(6)

thisExp.saveAsWideText(filename + '.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
thisExp.abort()
win.close()
core.quit()
