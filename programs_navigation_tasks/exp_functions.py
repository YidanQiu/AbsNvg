import pandas as pd
from numpy.random import random, randint, normal, shuffle, choice as randchoice
from psychopy import visual, event, core, gui, data, logging, clock
from psychopy.hardware import keyboard
import os, time


def initial_exp(expName, session, unit, win_size, fullscreen=False):
    '''
    use this function to initialize the experiment
    :param expName: any string to specific the task
    :param session: any string to specific the session
    :param unit: 'cm' / 'height' ...
    :param win_size: a tuple to specific the width and height of the screen
    :param fullscreen: True / False
    :return:
    '''
    _thisDir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(_thisDir)
    psychopyVersion = '2021.2.3'
    expName = expName
    expInfo = {'participant': '', 'session': session}
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['psychopyVersion'] = psychopyVersion
    filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    thisExp = data.ExperimentHandler(name=expName, version='', extraInfo=expInfo, runtimeInfo=None,
                                     originPath='program.py', savePickle=True, saveWideText=True, dataFileName=filename)
    logFile = logging.LogFile(filename + '.log', level=logging.EXP)
    endExpNow = False
    frameTolerance = 0.001
    win = visual.Window(size=win_size, fullscr=fullscreen, screen=0, winType='pyglet', allowGUI=True,
                        allowStencil=False,
                        monitor='testMonitor', color=[0, 0, 0], colorSpace='rgb', blendMode='avg', useFBO=True,
                        units=unit)
    expInfo['frameRate'] = win.getActualFrameRate()
    if expInfo['frameRate'] != None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0
    defaultKeyboard = keyboard.Keyboard()
    return thisExp, expInfo, win, defaultKeyboard, filename, endExpNow


def make_distribute(p):
    '''input probability (0.1)'''
    p_hundred = p * 100
    p_list = [0] * 100
    for i in range(int(p_hundred)):
        p_list[i] = 1
    this_chosen = randchoice(p_list)
    return this_chosen


def show_stimuli(stimuli):
    for i in stimuli:
        i.draw()


def mkImg(win, file_name, size=(33.87, 19.05), pos=(0., 0.)):
    img = visual.ImageStim(win=win, name='img', image=file_name, mask=None, ori=0.0, pos=pos, size=size,
                           color=[1, 1, 1], colorSpace='rgb', opacity=None, flipHoriz=False, flipVert=False,
                           texRes=128.0, interpolate=True, depth=0.0)
    return img


def mkText(win, this_text, font, pos, height, color):
    text = visual.TextStim(win=win, name='text', text=this_text, font=font,
                           pos=pos, height=height, wrapWidth=None, ori=0.0, color=color,
                           colorSpace='rgb', opacity=None, languageStyle='LTR', depth=-1.0)
    return text


def run_introduction(win, thisExp, ins, glob_time, confirm_button='space'):
    ins.draw()
    intro_start = time.time() - glob_time
    win.flip()
    intro_resp = event.waitKeys(keyList=[confirm_button, 'escape'])
    if intro_resp[0] == 'escape':
        win.close()
        core.quit()
    else:
        intro_stop = time.time() - glob_time
        thisExp.addData('intro_start', intro_start)
        thisExp.addData('intro_stop', intro_stop)


def initialise_D1(space_shape, stimuliList, buttons):
    space_list = []
    cur = None
    goal = None
    corr_resp = None
    for i in range(space_shape[0]):
        space_list.append(i)
    not_satisfied = True
    while not_satisfied:
        cur, goal = randchoice(space_list, 2, replace=False)
        if cur in [space_list[0], space_list[-1]]:
            not_satisfied = True
        else:
            not_satisfied = False
    # print('current shape:', cur, '; goal shape:', goal)
    if goal > cur:
        corr_resp = buttons[1]
    elif goal < cur:
        corr_resp = buttons[0]
    cur_file = stimuliList[cur]
    goal_file = stimuliList[goal]
    choiceF_file = stimuliList[cur - 1]
    choiceJ_file = stimuliList[cur + 1]
    return [cur_file, goal_file, choiceF_file, choiceJ_file, corr_resp], cur, goal


def updateRoute_D1(file_list, cur, goal, stimuliList, buttons):
    cur_file, goal_file, choiceF_file, choiceJ_file, corr_resp = file_list
    cur_file = stimuliList[cur]
    if cur + 1 > 5:
        corr_resp = buttons[0]
        choiceJ_file = r'materials\intro_imgs\blank.png'
        buttons.remove(buttons[1])
        choiceF_file = stimuliList[cur - 1]
    elif cur - 1 < 0:
        corr_resp = buttons[1]
        choiceF_file = r'materials\intro_imgs\blank.png'
        buttons.remove(buttons[0])
        choiceJ_file = stimuliList[cur + 1]
    else:
        if cur > goal:
            corr_resp = buttons[0]
        if cur < goal:
            corr_resp = buttons[1]
        choiceF_file = stimuliList[cur - 1]
        choiceJ_file = stimuliList[cur + 1]
    file_list = [cur_file, goal_file, choiceF_file, choiceJ_file, corr_resp]
    return file_list, buttons


def findShortest(choiceSpace, goal):
    temp_list = []
    for num in choiceSpace:
        temp_list.append(abs(num - goal))
    index_best = temp_list.index(min(temp_list))
    best = choiceSpace[index_best]
    return best


def checkChoice(check, boundary):
    checked = check.copy()
    for this_check in checked:
        if this_check >= boundary or this_check < 0:
            checked.remove(this_check)
    return checked


def initialise_D3(space_shape):
    cur = []
    goal = None
    for i in range(len(space_shape)):
        cur.append(randchoice(list(range(space_shape[i]))))
    goalUnset = True  # 不跟起点重复
    while goalUnset:
        goal = []
        for i in range(len(space_shape)):
            goal.append(randchoice(list(range(space_shape[i]))))
        if goal != cur:
            goalUnset = False
    return cur, goal


def follow_D3(material, session, trial_type, cur, goal, dim_list, space_shape, keyList):
    '''
    :param material: excel file storing coordinate and the corresponding picture files.
    :param session: directory name of the picture files.
    :param trial_type: "normal" or "block"
    :param cur: current location coordinate
    :param goal: goal location coordinate
    :param dim_list: title of each dimension in the material excel
    :param space_shape: shape of the space
    :param keyList: ['d','f','j','k']
    :return: [cur_file, goal_file, choiceD_file, choiceF_file, choiceJ_file, choiceK_file, bestChoice_file,
             corr_resp, trialType], [cur, goal, bestChoice], finalChoices
    '''
    material_list = data.importConditions(material)
    space_a, space_b, space_c = dim_list
    cur_file = None
    goal_file = None
    choiceD_file = None
    choiceF_file = None
    choiceJ_file = None
    choiceK_file = None
    bestChoice_file = None

    cur_a, cur_b, cur_c = cur
    goal_a, goal_b, goal_c = goal

    # 相邻选项
    choiceSpace_a = [cur_a - 1, cur_a, cur_a + 1]
    choiceSpace_b = [cur_b - 1, cur_b, cur_b + 1]
    choiceSpace_c = [cur_c - 1, cur_c, cur_c + 1]

    # 去掉超出空间的点
    choiceSpace_a = checkChoice(choiceSpace_a, space_shape[0])
    choiceSpace_b = checkChoice(choiceSpace_b, space_shape[1])
    choiceSpace_c = checkChoice(choiceSpace_c, space_shape[2])

    best_a = findShortest(choiceSpace_a, goal_a)
    best_b = findShortest(choiceSpace_b, goal_b)
    best_c = findShortest(choiceSpace_c, goal_c)
    bestChoice = [best_a, best_b, best_c]

    choice_list = []
    for a in choiceSpace_a:
        for b in choiceSpace_b:
            for c in choiceSpace_c:
                choice_list.append([a, b, c])
    choice_list.remove(cur)  # 删掉跟 cur 重复的点
    choice_list.remove(bestChoice)  # 删掉最佳选择，方便后面随机选 26-1=25 个点

    if trial_type == 'normal':
        selectedChoices = [bestChoice]
        selectedChoices_index = randchoice(list(range(len(choice_list))), 3, replace=False)
        for i in selectedChoices_index:
            selectedChoices.append(choice_list[i])
    else:
        selectedChoices = []
        selectedChoices_index = randchoice(list(range(len(choice_list))), 4, replace=False)
        for i in selectedChoices_index:
            selectedChoices.append(choice_list[i])

    # 整理选项顺序
    sortRule = [[cur_a, cur_b + 1, cur_c - 1], [cur_a, cur_b, cur_c - 1], [cur_a, cur_b - 1, cur_c - 1],
                [cur_a - 1, cur_b + 1, cur_c - 1], [cur_a - 1, cur_b, cur_c - 1], [cur_a - 1, cur_b - 1, cur_c - 1],
                [cur_a - 1, cur_b + 1, cur_c], [cur_a - 1, cur_b, cur_c], [cur_a - 1, cur_b - 1, cur_c],
                [cur_a - 1, cur_b + 1, cur_c + 1], [cur_a - 1, cur_b, cur_c + 1], [cur_a - 1, cur_b - 1, cur_c + 1],
                [cur_a, cur_b + 1, cur_c + 1], [cur_a, cur_b, cur_c + 1], [cur_a, cur_b - 1, cur_c + 1],
                [cur_a, cur_b + 1, cur_c], [cur_a, cur_b - 1, cur_c],
                [cur_a + 1, cur_b + 1, cur_c + 1], [cur_a + 1, cur_b, cur_c + 1], [cur_a + 1, cur_b - 1, cur_c + 1],
                [cur_a + 1, cur_b + 1, cur_c], [cur_a + 1, cur_b, cur_c], [cur_a + 1, cur_b - 1, cur_c],
                [cur_a + 1, cur_b + 1, cur_c - 1], [cur_a + 1, cur_b, cur_c - 1], [cur_a + 1, cur_b - 1, cur_c - 1]]

    order = []
    for i in selectedChoices:
        order.append(sortRule.index(i))

    sortedChoices = [[0, 0]] * len(sortRule)
    for i in range(len(sortRule)):
        try:
            orIndex = order.index(i)
            sortedChoices[i] = selectedChoices[orIndex]
        except:
            sortedChoices[i] = [0, 0]

    toClear = [0, 0] in sortedChoices
    while toClear:
        sortedChoices.remove([0, 0])
        toClear = [0, 0] in sortedChoices

    finalChoices = sortedChoices
    choiceD, choiceF, choiceJ, choiceK = finalChoices

    # 正确答案
    bestIndex = finalChoices.index(bestChoice)
    corr_resp = keyList[bestIndex]

    for i in range(len(material_list)):
        if material_list[i][space_a] == cur_a and material_list[i][space_b] == cur_b and material_list[i][
            space_c] == cur_c:
            cur_file = rf"materials\{session}\{material_list[i]['file']}.png"
        if material_list[i][space_a] == goal_a and material_list[i][space_b] == goal_b and material_list[i][
            space_c] == goal_c:
            goal_file = rf"materials\{session}\{material_list[i]['file']}.png"
        if material_list[i][space_a] == best_a and material_list[i][space_b] == best_b and material_list[i][
            space_c] == best_c:
            bestChoice_file = rf"materials\{session}\{material_list[i]['file']}.png"

        if material_list[i][space_a] == choiceD[0] and material_list[i][space_b] == choiceD[1] and material_list[i][
            space_c] == choiceD[2]:
            choiceD_file = rf"materials\{session}\{material_list[i]['file']}.png"
        if material_list[i][space_a] == choiceF[0] and material_list[i][space_b] == choiceF[1] and material_list[i][
            space_c] == choiceF[2]:
            choiceF_file = rf"materials\{session}\{material_list[i]['file']}.png"
        if material_list[i][space_a] == choiceJ[0] and material_list[i][space_b] == choiceJ[1] and material_list[i][
            space_c] == choiceJ[2]:
            choiceJ_file = rf"materials\{session}\{material_list[i]['file']}.png"
        if material_list[i][space_a] == choiceK[0] and material_list[i][space_b] == choiceK[1] and material_list[i][
            space_c] == choiceK[2]:
            choiceK_file = rf"materials\{session}\{material_list[i]['file']}.png"

    return ([cur_file, goal_file, choiceD_file, choiceF_file, choiceJ_file, choiceK_file, bestChoice_file,
             corr_resp], bestChoice, finalChoices)


def follow_D2(material, session, trial_type, cur, goal, dim_list, space_shape, keyList):
    material_list = data.importConditions(material)
    space_a, space_b = dim_list
    cur_file = None
    goal_file = None
    choices_file = [None, None, None, None]
    bestChoice_file = None
    corr_resp = None

    cur_a, cur_b = cur
    goal_a, goal_b = goal

    # 相邻选项
    choiceSpace_a = [cur_a - 1, cur_a, cur_a + 1]
    choiceSpace_b = [cur_b - 1, cur_b, cur_b + 1]

    # 去掉超出空间的点
    choiceSpace_a = checkChoice(choiceSpace_a, space_shape[0])
    choiceSpace_b = checkChoice(choiceSpace_b, space_shape[1])

    best_a = findShortest(choiceSpace_a, goal_a)
    best_b = findShortest(choiceSpace_b, goal_b)
    bestChoice = [best_a, best_b]

    choice_list = []
    for a in choiceSpace_a:
        for b in choiceSpace_b:
            choice_list.append([a, b])
    choice_list.remove(cur)  # 删掉跟 cur 重复的点
    choice_list.remove(bestChoice)  # 删掉最佳选择，方便后面随机选

    if len(choice_list) < 3:  # corner, only 3 choices
        if cur_a == 0:
            if cur_b == 0:
                sortedChoices = [['blank', 'blank'], [cur_a, cur_b + 1], [cur_a + 1, cur_b + 1], [cur_a + 1, cur_b]]
                buttons = [keyList[1], keyList[2], keyList[3]]
            else:
                sortedChoices = [[cur_a, cur_b - 1], ['blank', 'blank'], [cur_a + 1, cur_b], [cur_a + 1, cur_b - 1]]
                buttons = [keyList[0], keyList[2], keyList[3]]
        else:
            if cur_b == 0:
                sortedChoices = [[cur_a - 1, cur_b], [cur_a - 1, cur_b + 1], [cur_a, cur_b + 1], ['blank', 'blank']]
            else:
                sortedChoices = [[cur_a, cur_b - 1], [cur_a - 1, cur_b - 1], [cur_a - 1, cur_b], ['blank', 'blank']]
            buttons = [keyList[0], keyList[1], keyList[2]]

        if trial_type == 'block':
            for this_choice in range(len(sortedChoices)):
                if sortedChoices[this_choice] == bestChoice:
                    sortedChoices[this_choice] = 'blank'
                    temp_best = keyList[this_choice]
                    buttons.remove(temp_best)

    else:
        buttons = keyList
        if trial_type == 'normal':
            selectedChoices = [bestChoice]
            selectedChoices_index = randchoice(list(range(len(choice_list))), 3, replace=False)
            for i in selectedChoices_index:
                selectedChoices.append(choice_list[i])
        else:
            selectedChoices = []
            selectedChoices_index = randchoice(list(range(len(choice_list))), 4, replace=False)
            for i in selectedChoices_index:
                selectedChoices.append(choice_list[i])
        # 整理选项顺序
        sortRule = [[cur_a, cur_b - 1], [cur_a - 1, cur_b - 1], [cur_a - 1, cur_b], [cur_a - 1, cur_b + 1],
                    [cur_a, cur_b + 1], [cur_a + 1, cur_b + 1], [cur_a + 1, cur_b], [cur_a + 1, cur_b - 1]]

        order = []
        for i in selectedChoices:
            order.append(sortRule.index(i))

        sortedChoices = [[0]] * len(sortRule)
        for i in range(len(sortRule)):
            try:
                orIndex = order.index(i)
                sortedChoices[i] = selectedChoices[orIndex]
            except:
                sortedChoices[i] = [0]

        toClear = [0] in sortedChoices
        while toClear:
            sortedChoices.remove([0])
            toClear = [0] in sortedChoices

    finalChoices = sortedChoices
    for i in range(len(finalChoices)):
        if 'blank' in finalChoices[i]:
            choices_file[i] = r'materials/intro_imgs/blank.png'

    # 正确答案
    if trial_type == 'normal':
        bestIndex = finalChoices.index(bestChoice)
        corr_resp = keyList[bestIndex]
    for i in range(len(material_list)):
        if material_list[i][space_a] == cur_a and material_list[i][space_b] == cur_b:
            cur_file = rf"materials\{session}\{material_list[i]['file']}.png"
        if material_list[i][space_a] == goal_a and material_list[i][space_b] == goal_b:
            goal_file = rf"materials\{session}\{material_list[i]['file']}.png"
        if material_list[i][space_a] == best_a and material_list[i][space_b] == best_b:
            bestChoice_file = rf"materials\{session}\{material_list[i]['file']}.png"
        for j in range(len(finalChoices)):
            if material_list[i][space_a] == finalChoices[j][0] and material_list[i][space_b] == finalChoices[j][1]:
                choices_file[j] = rf"materials\{session}\{material_list[i]['file']}.png"
    choiceD_file, choiceF_file, choiceJ_file, choiceK_file = choices_file

    return ([cur_file, goal_file, choiceD_file, choiceF_file, choiceJ_file, choiceK_file, bestChoice_file,
             corr_resp], bestChoice, finalChoices, buttons)


if __name__ == '__main__':
    pass
