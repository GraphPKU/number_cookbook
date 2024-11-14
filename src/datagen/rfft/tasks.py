# The script is used to generate datasets.

import numpy as np
np.random.seed(42)
import random
# random.seed(42)
from datagen.rfft.prompt import Prompt
from copy import deepcopy
import re
# from ..task import Task

    
def find_dot(result_int: str, pos: int, name_a: str) -> str:
    P = Prompt('find_dot')
    output = P.initialize.format(result_int, pos)
    cnt = 0
    result_dec = ''
    while cnt < pos:
        output += P.enter
        last_digit = result_int[-1] if result_int else '0'
        output += P.last_digit.format(last_digit)
        output += P.update_result.format(last_digit, result_dec, last_digit + result_dec, result_int[:-1])
        result_dec = last_digit + result_dec
        result_int = result_int[:-1]
        output += P.update_cnt.format(cnt, cnt+1)
        cnt += 1
    output += P.not_enter
    result_int = result_int.lstrip('0') or '0'
    result_dec = result_dec.rstrip('0') or '0'
    output += P.strip.format(result_int, result_dec)
    result = result_int + '.' + result_dec
    output += P.merge.format(result_int, result_dec, result)
    output += P.ret.format(result)
    output = output.replace('_var_res_', name_a)
    return output, result

def get_digit_Integer_int_int(num: str, pos: str) -> int:
    pos = int(pos)
    P = Prompt('get_digit_Integer_int_int')
    output = P.initialize.format(num, pos)
    cnt = 0
    while cnt < pos:
        output += P.enter
        output += P.update_num.format(num[1:])
        num = num[1:]
        output += P.update_cnt.format(cnt, cnt+1)
        cnt += 1
    output += P.not_enter
    output += P.ret.format(num[0])
    return output, num[0]

def get_digit_Float_int_int(num: str, pos: str) -> int:
    pos = int(pos)
    P = Prompt('get_digit_Float_int_int')
    output = P.initialize.format(num, pos)
    cnt = 0
    num = num.replace('.', '') 
    output += P.remove_dot.format(num)
    while cnt < pos:
        output += P.enter
        output += P.update_num.format(num[1:])
        num = num[1:]
        output += P.update_cnt.format(cnt, cnt+1)
        cnt += 1
    output += P.not_enter
    output += P.ret.format(num[0])
    return output, num[0]
    
    
def add_integer_integer_integer(num1: str, num2: str, name_a: str, name_b: str, name_c: str) -> str:
    P = Prompt('add_Integer_Integer_Integer')
    output = P.initialize.format(num1, num2)
    result = ''
    carry = 0
    while num1 or num2:
        output += P.enter
        output += P.last_digit.format(num1, num2, num1[-3:] if num1 else 0, num2[-3:] if num2 else 0)
        digit1 = int(num1[-3:]) if num1 else 0
        digit2 = int(num2[-3:]) if num2 else 0
        output += P.sum.format(carry, digit1, digit2, carry, digit1 + digit2 + carry)
        total = digit1 + digit2 + carry
        output += P.update_result.format(total, total%1000, ('00' + str(total%1000))[-3:], result, ('00' + str(total%100))[-3:] + result, carry, total//1000)
        result = ('00' + str(total%1000))[-3:] + result
        carry = total//1000
        output += P.update_nums.format(num1[:-3] if num1 else num1, num2[:-3] if num2 else num2)
        num1 = num1[:-3] if num1 else num1
        num2 = num2[:-3] if num2 else num2
    output += P.not_enter

    if carry:
        output += P.carry_true.format(carry, str(carry), result, str(carry) + result)
        result = str(carry) + result
    else:
        output += P.carry_false.format(carry)
    result = result.lstrip('0') or '0'
    output += P.lstrip.format(result)
    output += P.ret.format(result)
    output = output.replace('_var1_', name_a).replace('_var2_', name_b).replace('_var_res_', name_c)
    return output, result


def add_float_float_float(num1: str, num2: str, name_a: str, name_b: str, name_c: str) -> str:
    P = Prompt('add_Float_Float_Float')
    int1, dec1 = num1.split('.')
    int2, dec2 = num2.split('.')
    len1 = len(dec1)
    len2 = len(dec2)
    output = P.initialize.format(num1, num2, int1, dec1, int2, dec2, len1, len2)
    output += P.condition1.format(len1, '<' if len1 < len2 else '>' if len1 > len2 else '==', len2)
    if len1 < len2:
        output += P.if1
        while len1 < len2:
            output += P.enter1
            output += P.update_vars1.format(dec1, dec1 + '0', len1, len1+1)
            dec1 += '0'
            len1 += 1
        output += P.out1
    elif len1 > len2:
        output += P.if2
        while len1 > len2:
            output += P.enter2
            output += P.update_vars2.format(dec2, dec2 + '0', len2, len2+1)
            dec2 += '0'
            len2 += 1
        output += P.out2
    else:
        output += P.not_branch
    full1 = int1 + dec1
    full2 = int2 + dec2
    output += P.full.format(int1, dec1, full1, int2, dec2, full2)
    
    output += P.add_integer
    new_output, result = add_integer_integer_integer(full1, full2, 'add1', 'add2', 'result')
    output += new_output
    output += P.exit_function.format(result, result)
    
    output += P.find_dot.format(result, len1)
    new_output, new_result = find_dot(result, int(len1), 'new_result')
    output += new_output
    output += P.exit_function2.format(new_result, new_result)
    output += P.ret.format(new_result)
    output = output.replace('_var1_', name_a).replace('_var2_', name_b).replace('_var_res_', name_c)
    return output, new_result
    
def sub_float_float_float(num1: str, num2: str, name_a: str, name_b: str, name_c: str) -> str:
    P = Prompt('sub_Float_Float_Float')
    int1, dec1 = num1.split('.')
    int2, dec2 = num2.split('.')
    len1 = len(dec1)
    len2 = len(dec2)
    output = P.initialize.format(num1, num2, int1, dec1, int2, dec2, len1, len2)
    output += P.condition1.format(len1, '<' if len1 < len2 else '>' if len1 > len2 else '==', len2)
    if len1 < len2:
        output += P.if1
        while len1 < len2:
            output += P.enter1
            output += P.update_vars1.format(dec1, dec1 + '0', len1, len1+1)
            dec1 += '0'
            len1 += 1
        output += P.out1
    elif len1 > len2:
        output += P.if2
        while len1 > len2:
            output += P.enter2
            output += P.update_vars2.format(dec2, dec2 + '0', len2, len2+1)
            dec2 += '0'
            len2 += 1
        output += P.out2
    else:
        output += P.not_branch
    full1 = int1 + dec1
    full2 = int2 + dec2
    output += P.full.format(int1, dec1, full1, int2, dec2, full2)
    
    output += P.sub_integer
    new_output, result = sub_integer_integer_integer(full1, full2, 'full1', 'full2', 'result')
    output += new_output
    output += P.exit_function.format(result, result)
    
    output += P.find_dot.format(result, len1)
    new_output, new_result = find_dot(result, int(len1), 'new_result')
    output += new_output
    output += P.exit_function2.format(new_result, new_result)
    output += P.ret.format(new_result)
    output = output.replace('_var1_', name_a).replace('_var2_', name_b).replace('_var_res_', name_c)
    return output, new_result

def sub_integer_integer_integer(num1: str, num2: str, name_a: str, name_b: str, name_c: str) -> str:
    P = Prompt('sub_Integer_Integer_Integer')
    output = P.initialize.format(num1, num2)
    result = ''
    borrow = 0
    while num1 or num2:
        output += P.enter
        output += P.last_digit.format(num1, num2, num1[-3:] if num1 else 0, num2[-3:] if num2 else 0)
        digit1 = int(num1[-3:]) if num1 else 0
        digit2 = int(num2[-3:]) if num2 else 0
        output += P.sub.format(borrow, digit1, digit2, borrow, digit1 - digit2 - borrow)
        total = digit1 - digit2 - borrow
        if total < 0:
            output += P.borrow_true.format(total, total, total+1000)
            total += 1000
            borrow = 1
        else:
            output += P.borrow_false.format(total)
            borrow = 0
        output += P.update_result.format(total, result, ('00' + str(total))[-3:] + result)
        result = ('00' + str(total))[-3:] + result

        output += P.update_nums.format(num1[:-3] if num1 else num1, num2[:-3] if num2 else num2)
        num1 = num1[:-3] if num1 else num1
        num2 = num2[:-3] if num2 else num2
    output += P.not_enter
    result = result.lstrip('0') or '0'
    output += P.lstrip.format(result)
    output += P.ret.format(result)
    output = output.replace('num1', name_a).replace('num2', name_b).replace('result', name_c)
    return output, result

def multiply_integer_integer_integer(num1: str, num2: str, name_a: str, name_b: str, name_c: str) -> str:
    P = Prompt('multiply_hard_Integer_Integer_Integer')
    output = P.initialize.format(num1, num2)
    result = 0
    base = 0
    while num1:
        output += P.enter
        output += P.last_digit.format(num1[-1])
        digit1 = int(num1[-1])
        temp = int(num2) * digit1
        output += P.multiply.format(num2, digit1, temp)
        
        # add temp to result
        output += P.update_temp.format(temp, base, temp * 10 ** base, base, base+1)
        temp *= 10 ** base
        base += 1
        output += P.add_temp.format(result, temp, result + temp)
        result += temp
        output += P.update_result_num.format(num1[:-1])
        num1 = num1[:-1]
    output += P.not_enter
    output += P.ret.format(result)
    output = output.replace('num1', name_a).replace('num2', name_b).replace('result', name_c)
    return output, str(result)

def multiply_float_float_float(num1: str, num2: str, name_a: str, name_b: str, name_c: str) -> str:
    P = Prompt('multiply_hard_Float_Float_Float')
    int1, dec1 = num1.split('.')
    int2, dec2 = num2.split('.')
    len1 = len(dec1)
    len2 = len(dec2)
    output = P.initialize.format(num1, num2, int1, dec1, int2, dec2, len1, len2)
    full1 = int1 + dec1
    full2 = int2 + dec2
    output += P.full.format(int1, dec1, full1, int2, dec2, full2)
    
    output += P.multiply_integer
    new_output, result = multiply_integer_integer_integer(full1, full2, 'mul1', 'mul2', 'result')
    output += new_output
    output += P.exit_function.format(result, result)
    
    output += P.find_dot.format(result, len1)
    new_output, new_result = find_dot(result, int(len1) + int(len2), 'new_result')
    output += new_output
    output += P.exit_function2.format(new_result, new_result)
    output += P.ret.format(new_result)
    output = output.replace('_var1_', name_a).replace('_var2_', name_b).replace('_var_res_', name_c)
    return output, new_result


def digit_max_integer_integer_integer(num1: str, num2: str) -> str:
    P = Prompt('digit_max_Integer_Integer_Integer')
    output = P.initialize.format(num1, num2)
    result = ''
    output = ''
    while num1 and num2:
        output += P.enter
        output += P.last_digit.format(num1, num2, num1[-1], num2[-1])
        digit1 = num1[-1]
        digit2 = num2[-1]
        output += P.max.format(max(digit1, digit2), digit1, digit2, max(digit1, digit2) + result)
        result = str(max(digit1, digit2)) + result
        output += P.update_nums.format(num1[:-1], num2[:-1])
        num1 = num1[:-1]
        num2 = num2[:-1]
    output += P.not_enter
    output += P.update_rest.format(num1, num2, result, num1 + num2 + result)
    result = num1 + num2 + result
    output += P.ret.format(result)
    return output, result

def length_Integer_none_int(num: str) -> int:
    P = Prompt('length_Integer_none_int')
    output = P.initialize.format(num)
    result = 0
    output = ''
    while num:
        output += P.enter
        output += P.update_result.format(result, result+1)
        result += 1
        output += P.update_nums.format(num[1:])
        num = num[1:]
    output += P.not_enter
    output += P.ret.format(result)
    return output, result

def floordiv_integer_integer_integer(num1: str, num2: str, name_a: str, name_b: str, name_c: str) -> str:
    P = Prompt('floordiv_Integer_Integer_Integer')
    output = P.initialize.format(num1, num2)
    num2 = int(num2)
    result = ''
    num_now = 0
    while num1:
        output += P.enter
        output += P.update_num_now.format(num_now, num1[0], num_now * 10 + int(num1[0]))
        num_now = num_now * 10 + int(num1[0])
        output += P.digit_init
        digit = 0
        while num_now and int(num_now) >= int(num2):
            output += P.enter_digit
            output += P.update_num_now2.format(num_now, num2, num_now-num2)
            num_now -= num2
            output += P.update_digit.format(digit, digit+1)
            digit += 1
        output += P.not_enter_digit
        output += P.update_result.format(result, str(digit), result + str(digit))
        result += str(digit)
        output += P.update_nums.format(num1[1:])
        num1 = num1[1:]
    output += P.not_enter
    result = result.lstrip('0') or '0'
    output += P.lstrip.format(result)
    output += P.ret.format(result)
    output = output.replace('_var1_', name_a).replace('_var2_', name_b).replace('_var_res_', name_c)
    return output, result

def brief_floordiv_integer_integer_integer(num1: str, num2: str, name_a: str, name_b: str, name_c: str) -> str:
    P = Prompt('floordiv_Integer_Integer_Integer')
    output = P.initialize.format(num1, num2)
    num2 = int(num2)
    result = ''
    num_now = 0
    while num1:
        output += P.enter
        output += P.update_num_now.format(num_now, num1[0], num_now * 10 + int(num1[0]))
        num_now = num_now * 10 + int(num1[0])
        output += P.brief.format(num_now, num2, num_now // num2, num_now, num2, num_now % num2)
        digit = num_now // num2
        num_now = num_now % num2
        output += P.update_result.format(result, str(digit), result + str(digit))
        result += str(digit)
        output += P.update_nums.format(num1[1:])
        num1 = num1[1:]
    output += P.not_enter
    result = result.lstrip('0') or '0'
    output += P.lstrip.format(result)
    output += P.ret.format(result)
    output = output.replace('_var1_', name_a).replace('_var2_', name_b).replace('_var_res_', name_c)
    return output, result

def mod_integer_integer_integer(num1: str, num2: str, name_a: str, name_b: str) -> str:
    P = Prompt('mod_Integer_Integer_Integer')
    output = P.initialize.format(num1, num2)
    
    output += P.calc_q
    new_output, q = brief_floordiv_integer_integer_integer(num1, num2, 'div1', 'div2', 'q')
    output += new_output
    output += P.exit_function1.format(q)
    
    output += P.calc_temp
    new_output, temp = multiply_integer_integer_integer(q, num2, 'mul1', 'mul2', 'temp')
    output += new_output
    output += P.exit_function2.format(temp)
    
    output += P.calc_r
    new_output, r = sub_integer_integer_integer(num1, temp, 'sub1', 'sub2', 'r')
    output += new_output
    output += P.exit_function3.format(r)
    
    output += P.ret.format(r)
    output = output.replace('_var1_', name_a).replace('_var2_', name_b)
    return output, r
    
def gcd_integer_integer_integer(num1: str, num2: str, name_a: str, name_b: str) -> str:
    output = '''the greatest common divisor of {} and {} is {}
'''.format(num1, num2, np.gcd(int(num1), int(num2)))
    return output, str(np.gcd(int(num1), int(num2)))

class Dataset_Generator:
    def __init__(self, dataset_name: str) -> None:
        self.name = dataset_name
        
    def rfft_IO(self, q_str) -> dict:
        '''
        return rfft input-output of given data
        '''
        P = Prompt(self.name)
        rule = P.rule
        instruction = "Follow the given rule to solve the question.\nrule:\n"
        if self.name == "add_Integer_Integer_Integer":
            num1, num2, expected = re.findall(r'(-?\d+)', q_str.split(':')[-1])
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            input = input.replace('_var1_', 'num1').replace('_var2_', 'num2').replace('_var_res_', 'result')
            # rfft output
            output, answer = add_integer_integer_integer(num1, num2, 'num1', 'num2', 'result')
            output += P.answer.format(answer)
            assert answer == expected, f'expected: {expected}, got: {answer}'
        
        if self.name == 'add_Fraction_Fraction_Fraction':
            # 用正则表达式提取全部形如1/2分数表示的数字
            num1, num2, expected = re.findall(r'(-?\d+/\d+)', q_str)[-3:]
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            
            # rfft output
            frac1, frac2 = num1, num2
            num1, den1 = frac1.split('/')
            num2, den2 = frac2.split('/')
            output = P.initialize.format(frac1, frac2, num1, den1, num2, den2)
            
            output += P.calc_den
            new_output, den = multiply_integer_integer_integer(den1, den2, 'mul1', 'mul2', 'den')
            output += new_output
            output += P.exit_function1.format(den, den)
            
            output += P.calc_tmp1
            new_output, tmp1 = multiply_integer_integer_integer(num1, den2, 'mul1', 'mul2', 'tmp1')
            output += new_output
            output += P.exit_function2.format(tmp1, tmp1)
            
            output += P.calc_tmp2
            new_output, tmp2 = multiply_integer_integer_integer(num2, den1, 'mul1', 'mul2', 'tmp2')
            output += new_output
            output += P.exit_function3.format(tmp2, tmp2)
            
            output += P.add_num
            new_output, num = add_integer_integer_integer(tmp1, tmp2, 'add1', 'add2', 'num')
            output += new_output
            output += P.exit_function4.format(num, num)
            
            output += P.calc_gcd
            new_output, gcd = gcd_integer_integer_integer(num, den, 'gcd1', 'gcd2')
            output += new_output
            output += P.exit_function5.format(gcd, gcd)
            
            output += P.calc_num2
            new_output, num = brief_floordiv_integer_integer_integer(num, gcd, 'div1', 'div2', 'num')
            output += new_output
            output += P.exit_function6.format(num, num)
            
            output += P.calc_den2
            new_output, den = brief_floordiv_integer_integer_integer(den, gcd, 'div1', 'div2', 'den')
            output += new_output
            output += P.exit_function7.format(den, den)
            
            result = num + '/' + den
            output += P.calc_result.format(num, den, result)
            answer = result
            output += P.answer.format(answer)
            assert answer == expected, f'expected: {expected}, got: {answer}'
        
        if self.name == 'add_ScientificNotation_ScientificNotation_ScientificNotation':
            # 用正则表达式提取全部形如1.1e123科学计数法表示的数字
            num1, num2, expected = re.findall(r'(-?\d+\.\d+e-?\d+)', q_str)[-3:]
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            # rfft output
            output = P.initialize.format(num1, num2)
            base1, exp1 = num1.split('e')
            base2, exp2 = num2.split('e')
            exp1, exp2 = int(exp1), int(exp2)
            output += P.split.format(base1, exp1, base2, exp2)
            output += P.condition.format(exp1, '<' if exp1 < exp2 else '>' if exp1 > exp2 else '==', exp2)
            if exp1 < exp2:
                base1, base2 = base2, base1
                exp1, exp2 = exp2, exp1
                output += P.if1.format(base1, base2, exp1, exp2)
            else:
                output += P.not_branch
            int1, dec1 = base1.split('.')
            output += P.split2.format(int1, dec1, exp1, exp2, exp1 - exp2)
            cnt = exp1 - exp2
            while cnt > 0:
                output += P.enter
                first_digit = dec1[0] if dec1 else '0'
                output += P.update_nums.format(int1, first_digit, int1 + first_digit, dec1[1:], cnt, cnt-1)
                int1 += dec1[0] if dec1 else '0'
                dec1 = dec1[1:] if dec1 else dec1
                cnt -= 1
            output += P.out
            if dec1 == '':
                dec1 = '0'
            output += P.update_dec.format(dec1)
            base1 = int1 + '.' + dec1
            output += P.update_base1.format(int1, dec1, base1)
            result_exp = exp2
            output += P.update_exp.format(exp2)
            output += P.add_float
            new_output, result_base = add_float_float_float(base1, base2, 'add1', 'add2', 'result_base')
            output += new_output
            output += P.exit_function1.format(result_base)
            
            int_part, dec_part = result_base.split('.')
            output += P.split3.format(int_part, dec_part)
            while len(int_part) > 1:
                output += P.enter2
                output += P.update_parts.format(int_part[-1], dec_part, int_part[-1] + dec_part, int_part[:-1])
                dec_part = int_part[-1] + dec_part
                int_part = int_part[:-1]
                output += P.update_exp2.format(result_exp, result_exp+1)
                result_exp += 1
            output += P.out2
            dec_part = dec_part.rstrip('0') or '0'
            output += P.rstrip.format(dec_part)
            result_base = int_part + '.' + dec_part
            output += P.update_result_base.format(int_part, dec_part, result_base)
            result = result_base + 'e' + str(result_exp)
            output += P.merge.format(result_base, result_exp, result)
            output += P.ret.format(result)
            answer = result
            output += P.answer.format(answer)
            assert answer == expected, f'expected: {expected}, got: {answer}'
            
        if self.name == 'add_Float_Float_Float':

            num1, num2, expected = re.findall(r'(-?\d+\.\d+)', q_str)[-3:]
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            input = input.replace('_var1_', 'num1').replace('_var2_', 'num2').replace('_var_res_', 'result')
            # rfft output
            output, answer = add_float_float_float(num1, num2, 'num1', 'num2', 'result')
            # output += P.ret.format(answer)
            output += P.answer.format(answer)
            assert answer == expected, f'expected: {expected}, got: {answer}'
        
        if self.name == 'add_Fraction_Fraction_Fraction' or self.name == 'add_easy_Fraction_Fraction_Fraction':
            num1, num2, expected = re.findall(r'(-?\d+/\d+)', q_str)[-3:]
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            # rfft output
            frac1, frac2 = num1, num2
            num1, den1 = frac1.split('/')
            num2, den2 = frac2.split('/')
            output = P.initialize.format(frac1, frac2, num1, den1, num2, den2)
            
            output += P.calc_den
            new_output, den = multiply_integer_integer_integer(den1, den2, 'mul1', 'mul2', 'den')
            # output += new_output
            output += P.exit_function1.format(den, den)
            
            output += P.calc_tmp1
            new_output, tmp1 = multiply_integer_integer_integer(num1, den2, 'mul1', 'mul2', 'tmp1')
            # output += new_output
            output += P.exit_function2.format(tmp1, tmp1)
            
            output += P.calc_tmp2
            new_output, tmp2 = multiply_integer_integer_integer(num2, den1, 'mul1', 'mul2', 'tmp2')
            # output += new_output
            output += P.exit_function3.format(tmp2, tmp2)
            
            output += P.add_num
            new_output, num = add_integer_integer_integer(tmp1, tmp2, 'add1', 'add2', 'num')
            output += new_output
            output += P.exit_function4.format(num, num)
            
            output += P.calc_gcd
            new_output, gcd = gcd_integer_integer_integer(num, den, 'gcd1', 'gcd2')
            output += new_output
            output += P.exit_function5.format(gcd, gcd)
            
            output += P.calc_num2
            new_output, num = brief_floordiv_integer_integer_integer(num, gcd, 'div1', 'div2', 'num')
            output += new_output
            output += P.exit_function6.format(num, num)
            
            output += P.calc_den2
            new_output, den = brief_floordiv_integer_integer_integer(den, gcd, 'div1', 'div2', 'den')
            output += new_output
            output += P.exit_function7.format(den, den)
            
            result = num + '/' + den
            output += P.calc_result.format(num, den, result)
            answer = result
            output += P.answer.format(answer)
            assert answer == expected, f'expected: {expected}, got: {answer}'
        
        if self.name == "sub_Integer_Integer_Integer":
            num1, num2, expected = re.findall(r'(-?\d+)', q_str.split(':')[-1])
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            input = input.replace('_var1_', 'num1').replace('_var2_', 'num2').replace('_var_res_', 'result')
            # rfft output
            output, answer = sub_integer_integer_integer(num1, num2, 'num1', 'num2', 'result')
            output += P.answer.format(answer)
            assert answer == expected, f'{q_str}\nexpected: {expected}, got: {answer}'
        
        if self.name == 'sub_ScientificNotation_ScientificNotation_ScientificNotation':
            num1, num2, expected = re.findall(r'(-?\d+\.\d+e-?\d+)', q_str)[-3:]
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            # rfft output
            output = P.initialize.format(num1, num2)
            base1, exp1 = num1.split('e')
            base2, exp2 = num2.split('e')
            exp1, exp2 = int(exp1), int(exp2)
            output += P.split.format(base1, exp1, base2, exp2)
            int1, dec1 = base1.split('.')
            output += P.split2.format(int1, dec1, exp1, exp2, exp1 - exp2)
            cnt = exp1 - exp2
            while cnt > 0:
                output += P.enter
                first_digit = dec1[0] if dec1 else '0'
                output += P.update_nums.format(int1, first_digit, int1 + first_digit, dec1[1:], cnt, cnt-1)
                int1 += dec1[0] if dec1 else '0'
                dec1 = dec1[1:] if dec1 else dec1
                cnt -= 1
            output += P.out
            if dec1 == '':
                dec1 = '0'
            output += P.update_dec.format(dec1)
            base1 = int1 + '.' + dec1
            output += P.update_base1.format(int1, dec1, base1)
            result_exp = exp2
            output += P.update_exp.format(exp2)
            output += P.sub_float
            new_output, result_base = sub_float_float_float(base1, base2, 'sub1', 'sub2', 'result_base')
            output += new_output
            output += P.exit_function1.format(result_base)
            
            int_part, dec_part = result_base.split('.')
            output += P.split3.format(int_part, dec_part)
            while len(int_part) > 1:
                output += P.enter2
                output += P.update_parts.format(int_part[-1], dec_part, int_part[-1] + dec_part, int_part[:-1])
                dec_part = int_part[-1] + dec_part
                int_part = int_part[:-1]
                output += P.update_exp2.format(result_exp, result_exp+1)
                result_exp += 1
            output += P.out2
            while int_part[0] == '0':
                output += P.enter3
                output += P.update_parts2.format(dec_part[0], dec_part[1:])
                int_part = dec_part[0]
                dec_part = dec_part[1:]
                output += P.update_exp3.format(result_exp, result_exp-1)
                result_exp -= 1
            output += P.out3
            dec_part = dec_part.rstrip('0') or '0'
            output += P.rstrip.format(dec_part)
            result_base = int_part + '.' + dec_part
            output += P.update_result_base.format(int_part, dec_part, result_base)
            result = result_base + 'e' + str(result_exp)
            output += P.merge.format(result_base, result_exp, result)
            output += P.ret.format(result)
            answer = result
            output += P.answer.format(answer)
            assert answer == expected, f'expected: {expected}, got: {answer}'
        
        if self.name == 'sub_Float_Float_Float':
            num1, num2, expected = re.findall(r'(-?\d+\.\d+)', q_str)[-3:]
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            input = input.replace('_var1_', 'num1').replace('_var2_', 'num2').replace('_var_res_', 'result')
            # rfft output
            output, answer = sub_float_float_float(num1, num2, 'num1', 'num2', 'result')
            # output += P.ret.format(answer)
            output += P.answer.format(answer)
            assert answer == expected, f'expected: {expected}, got: {answer}'
       
        if self.name == 'sub_Fraction_Fraction_Fraction':
            num1, num2, expected = re.findall(r'(-?\d+/\d+)', q_str)[-3:]
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            # rfft output
            frac1, frac2 = num1, num2
            num1, den1 = frac1.split('/')
            num2, den2 = frac2.split('/')
            output = P.initialize.format(frac1, frac2, num1, den1, num2, den2)
            
            output += P.calc_den
            new_output, den = multiply_integer_integer_integer(den1, den2, 'mul1', 'mul2', 'den')
            output += new_output
            output += P.exit_function1.format(den, den)
            
            output += P.calc_tmp1
            new_output, tmp1 = multiply_integer_integer_integer(num1, den2, 'mul1', 'mul2', 'tmp1')
            output += new_output
            output += P.exit_function2.format(tmp1, tmp1)
            
            output += P.calc_tmp2
            new_output, tmp2 = multiply_integer_integer_integer(num2, den1, 'mul1', 'mul2', 'tmp2')
            output += new_output
            output += P.exit_function3.format(tmp2, tmp2)
            
            output += P.sub_num
            new_output, num = sub_integer_integer_integer(tmp1, tmp2, 'sub1', 'sub2', 'num')
            output += new_output
            output += P.exit_function4.format(num, num)
            
            output += P.calc_gcd
            new_output, gcd = gcd_integer_integer_integer(num, den, 'gcd1', 'gcd2')
            output += new_output
            output += P.exit_function5.format(gcd, gcd)
            
            output += P.calc_num2
            new_output, num = brief_floordiv_integer_integer_integer(num, gcd, 'div1', 'div2', 'num')
            output += new_output
            output += P.exit_function6.format(num, num)
            
            output += P.calc_den2
            new_output, den = brief_floordiv_integer_integer_integer(den, gcd, 'div1', 'div2', 'den')
            output += new_output
            output += P.exit_function7.format(den, den)
            
            result = num + '/' + den
            output += P.calc_result.format(num, den, result)
            answer = result
            output += P.answer.format(answer)
            assert answer == expected, f'expected: {expected}, got: {answer}'
        
        if self.name == 'max_Integer_Integer_Integer' or self.name == 'max_hard_Integer_Integer_Integer':
            num1, num2, expected = re.findall(r'(-?\d+)', q_str.split(':')[-1])
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            # rfft output
            len1 = len(num1)
            len2 = len(num2)
            output = P.initialize.format(num1, num2, len1, len2)
            output += P.condition1.format(len1, '<' if len1 < len2 else '>' if len1 > len2 else '==', len2)
            if len1 < len2:
                output += P.if1.format(num2)
                answer = num2
            elif len1 > len2:
                output += P.if2.format(num1)
                answer = num1
            else:
                output += P.if3.format(num1, num2)
                num1_copy = num1[:]
                num2_copy = num2[:]
                _break = False
                while num1 and num2:
                    output += P.enter
                    output += P.first_digit.format(num1[0], num2[0])
                    digit1 = num1[0]
                    digit2 = num2[0]
                    op = '<' if digit1 < digit2 else '>' if digit1 > digit2 else '='
                    output += P.compare.format(digit1, op, digit2)
                    if digit1 < digit2:
                        output += P.if4.format(num2_copy)
                        answer = num2_copy
                        _break = True
                        break
                    elif digit1 > digit2:
                        output += P.if5.format(num1_copy)
                        answer = num1_copy
                        _break = True
                        break
                    output += P.update_nums.format(num1[1:], num2[1:])
                    num1 = num1[1:]
                    num2 = num2[1:]
                if not _break:
                    output += P.not_enter
                    output += P.ret.format(num1_copy)
                    answer = num1_copy
            output += P.answer.format(answer)
            assert answer == expected, f'expected: {expected}, got: {answer}'
             
        if self.name == 'max_Float_Float_Float' or self.name == 'max_hard_Float_Float_Float':
            num1, num2, expected = re.findall(r'(-?\d+\.\d+)', q_str)[-3:]
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            # rfft output
            int1, dec1 = num1.split('.')
            int2, dec2 = num2.split('.')
            len1 = len(dec1)
            len2 = len(dec2)
            output = P.initialize.format(num1, num2, int1, dec1, int2, dec2, len1, len2)
            output += P.pcondition1.format(len1, '<' if len1 < len2 else '>' if len1 > len2 else '==', len2)
            if len1 < len2:
                output += P.pif1
                while len1 < len2:
                    output += P.penter1
                    output += P.pupdate_vars1.format(dec1, dec1 + '0', len1, len1+1)
                    dec1 += '0'
                    len1 += 1
                output += P.pout1
            elif len1 > len2:
                output += P.pif2
                while len1 > len2:
                    output += P.penter2
                    output += P.pupdate_vars2.format(dec2, dec2 + '0', len2, len2+1)
                    dec2 += '0'
                    len2 += 1
                output += P.pout2
            else:
                output += P.pnot_branch
            full1 = int1 + dec1
            full2 = int2 + dec2
            len1 = len(full1)
            len2 = len(full2)
            output += P.pfull.format(int1, dec1, full1, int2, dec2, full2, len1, len2)
            
            
            output += P.condition1.format(len1, '<' if len1 < len2 else '>' if len1 > len2 else '==', len2)
            if len1 < len2:
                output += P.if1.format(num2)
                answer = num2
            elif len1 > len2:
                output += P.if2.format(num1)
                answer = num1
            else:
                output += P.if3.format(num1, num2)
                num1_copy = num1[:]
                num2_copy = num2[:]
                _break = False
                num1 = full1
                num2 = full2
                while num1 and num2:
                    output += P.enter
                    output += P.first_digit.format(num1[0], num2[0])
                    digit1 = num1[0]
                    digit2 = num2[0]
                    op = '<' if digit1 < digit2 else '>' if digit1 > digit2 else '='
                    output += P.compare.format(digit1, op, digit2)
                    if digit1 < digit2:
                        output += P.if4.format(num2_copy)
                        answer = num2_copy
                        _break = True
                        break
                    elif digit1 > digit2:
                        output += P.if5.format(num1_copy)
                        answer = num1_copy
                        _break = True
                        break
                    output += P.update_nums.format(num1[1:], num2[1:])
                    num1 = num1[1:]
                    num2 = num2[1:]
                if not _break:
                    output += P.not_enter
                    output += P.ret.format(num1_copy)
                    answer = num1_copy
            output += P.answer.format(answer)
            assert answer == expected, f'expected: {expected}, got: {answer}'
                    
        if self.name == 'max_Fraction_Fraction_Fraction':
            num1, num2, expected = re.findall(r'(-?\d+/\d+)', q_str)[-3:]
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            # rfft output
            frac1, frac2 = num1, num2
            num1, den1 = frac1.split('/')
            num2, den2 = frac2.split('/')
            output = P.initialize.format(frac1, frac2, num1, den1, num2, den2)
            
            output += P.calc_tmp1
            new_output, tmp1 = multiply_integer_integer_integer(num1, den2, 'mul1', 'mul2', 'tmp1')
            output += new_output
            output += P.exit_function2.format(tmp1, tmp1)
            
            output += P.calc_tmp2
            new_output, tmp2 = multiply_integer_integer_integer(num2, den1, 'mul1', 'mul2', 'tmp2')
            output += new_output
            output += P.exit_function3.format(tmp2, tmp2)
            
            len1 = len(tmp1)
            len2 = len(tmp2)
            output += P.calc_len.format(len1, len2)
            
            
            output += P.condition1.format(len1, '<' if len1 < len2 else '>' if len1 > len2 else '==', len2)
            if len1 < len2:
                output += P.if1.format(frac2)
                answer = frac2
            elif len1 > len2:
                output += P.if2.format(frac1)
                answer = frac1
            else:
                output += P.if3
                num1_copy = frac1
                num2_copy = frac2
                num1 = tmp1
                num2 = tmp2
                _break = False
                while num1 and num2:
                    output += P.enter
                    output += P.first_digit.format(num1[0], num2[0])
                    digit1 = num1[0]
                    digit2 = num2[0]
                    op = '<' if digit1 < digit2 else '>' if digit1 > digit2 else '='
                    output += P.compare.format(digit1, op, digit2)
                    if digit1 < digit2:
                        output += P.if4.format(num2_copy)
                        answer = num2_copy
                        _break = True
                        break
                    elif digit1 > digit2:
                        output += P.if5.format(num1_copy)
                        answer = num1_copy
                        _break = True
                        break
                    output += P.update_nums.format(num1[1:], num2[1:])
                    num1 = num1[1:]
                    num2 = num2[1:]
                if not _break:
                    output += P.not_enter
                    output += P.ret.format(num1_copy)
                    answer = num1_copy
            output += P.answer.format(answer)
            assert answer == expected, f'expected: {expected}, got: {answer}'
        
        if self.name == 'max_ScientificNotation_ScientificNotation_ScientificNotation' or self.name == 'max_hard_ScientificNotation_ScientificNotation_ScientificNotation':
            num1, num2, expected = re.findall(r'(-?\d+\.\d+e-?\d+)', q_str)[-3:]
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            # rfft output
            output = P.sinitialize.format(num1, num2)
            base1, exp1 = num1.split('e')
            base2, exp2 = num2.split('e')
            exp1, exp2 = int(exp1), int(exp2)
            output += P.ssplit.format(base1, exp1, base2, exp2)
            output += P.scondition.format(exp1, '<' if exp1 < exp2 else '>' if exp1 > exp2 else '==', exp2)
            if exp1 < exp2:
                output += P.sif1.format(num2)
                answer = num2
            elif exp1 > exp2:
                output += P.sif2.format(num1)
                answer = num1
            else:
                output += P.snot_branch
                
                num1_copy = num1
                num2_copy = num2
                num1 = base1
                num2 = base2
                int1, dec1 = num1.split('.')
                int2, dec2 = num2.split('.')
                len1 = len(dec1)
                len2 = len(dec2)
                output += P.initialize.format(num1_copy, num2_copy, num1, num2, int1, dec1, int2, dec2, len1, len2)
                output += P.pcondition1.format(len1, '<' if len1 < len2 else '>' if len1 > len2 else '==', len2)
                if len1 < len2:
                    output += P.pif1
                    while len1 < len2:
                        output += P.penter1
                        output += P.pupdate_vars1.format(dec1, dec1 + '0', len1, len1+1)
                        dec1 += '0'
                        len1 += 1
                    output += P.pout1
                elif len1 > len2:
                    output += P.pif2
                    while len1 > len2:
                        output += P.penter2
                        output += P.pupdate_vars2.format(dec2, dec2 + '0', len2, len2+1)
                        dec2 += '0'
                        len2 += 1
                    output += P.pout2
                else:
                    output += P.pnot_branch
                full1 = int1 + dec1
                full2 = int2 + dec2
                len1 = len(full1)
                len2 = len(full2)
                output += P.pfull.format(int1, dec1, full1, int2, dec2, full2, len1, len2)
                
                
                output += P.condition1.format(len1, '<' if len1 < len2 else '>' if len1 > len2 else '==', len2)
                if len1 < len2:
                    output += P.if1.format(num2_copy)
                    answer = num2_copy
                elif len1 > len2:
                    output += P.if2.format(num1_copy)
                    answer = num1_copy
                else:
                    output += P.if3
                    _break = False
                    num1 = full1
                    num2 = full2
                    while num1 and num2:
                        output += P.enter
                        output += P.first_digit.format(num1[0], num2[0])
                        digit1 = num1[0]
                        digit2 = num2[0]
                        op = '<' if digit1 < digit2 else '>' if digit1 > digit2 else '='
                        output += P.compare.format(digit1, op, digit2)
                        if digit1 < digit2:
                            output += P.if4.format(num2_copy)
                            answer = num2_copy
                            _break = True
                            break
                        elif digit1 > digit2:
                            output += P.if5.format(num1_copy)
                            answer = num1_copy
                            _break = True
                            break
                        output += P.update_nums.format(num1[1:], num2[1:])
                        num1 = num1[1:]
                        num2 = num2[1:]
                    if not _break:
                        output += P.not_enter
                        output += P.ret.format(num1_copy)
                        answer = num1_copy
            output += P.answer.format(answer)
            assert answer == expected, f'expected: {expected}, got: {answer}'
        
        if self.name == 'multiply_hard_Integer_Integer_Integer' or self.name == 'multiply_easy_Integer_Integer_Integer': 
            num1, num2, expected = re.findall(r'(-?\d+)', q_str.split(':')[-1])
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            input = input.replace('_var1_', 'num1').replace('_var2_', 'num2').replace('_var_res_', 'result')
            # rfft output
            output, answer = multiply_integer_integer_integer(num1, num2, 'num1', 'num2', 'result')
            output += P.answer.format(answer)
            assert answer == expected, f'{q_str}\nexpected: {expected}, got: {answer}'
            
        if self.name == 'multiply_hard_Float_Float_Float' or self.name == 'multiply_easy_Float_Float_Float':
            num1, num2, expected = re.findall(r'(-?\d+\.\d+)', q_str)[-3:]
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            input = input.replace('_var1_', 'num1').replace('_var2_', 'num2').replace('_var_res_', 'result')
            # rfft output
            output, answer = multiply_float_float_float(num1, num2, 'num1', 'num2', 'result')
            output += P.answer.format(answer)
            assert answer == expected, f'expected: {expected}, got: {answer}'
        
        if self.name == 'multiply_hard_ScientificNotation_ScientificNotation_ScientificNotation' or self.name == 'multiply_easy_ScientificNotation_ScientificNotation_ScientificNotation':
            num1, num2, expected = re.findall(r'(-?\d+\.\d+e-?\d+)', q_str)[-3:]
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            # rfft output
            base1, exp1 = num1.split('e')
            base2, exp2 = num2.split('e')
            output = P.initialize.format(num1, num2, base1, exp1, base2, exp2)
            output += P.multiply_float
            new_output, result_base = multiply_float_float_float(base1, base2, 'mul1', 'mul2', 'result_base')
            output += new_output
            output += P.exit_function.format(result_base)
            
            result_exp = int(exp1) + int(exp2)
            output += P.result_exp.format(exp1, exp2, result_exp)
            int_part, dec_part = result_base.split('.')
            output += P.split.format(int_part, dec_part)
            while len(int_part) > 1:
                output += P.enter
                output += P.update_parts.format(int_part[-1], dec_part, int_part[-1] + dec_part, int_part[:-1])
                dec_part = int_part[-1] + dec_part
                int_part = int_part[:-1]
                output += P.update_exp.format(result_exp, result_exp+1)
                result_exp += 1
            output += P.out
            dec_part = dec_part.rstrip('0') or '0'
            output += P.rstrip.format(dec_part)
            result_base = int_part + '.' + dec_part
            output += P.update_result_base.format(int_part, dec_part, result_base)
            result = result_base + 'e' + str(result_exp)
            output += P.merge.format(result_base, result_exp, result)
            output += P.ret.format(result)
            answer = result
            output += P.answer.format(answer)
            assert answer == expected, f'expected: {expected}, got: {answer}'
            
        if self.name == 'multiply_hard_Fraction_Fraction_Fraction' or self.name == 'multiply_easy_Fraction_Fraction_Fraction':
            num1, num2, expected = re.findall(r'(-?\d+/\d+)', q_str)[-3:]
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            # rfft output
            frac1, frac2 = num1, num2
            num1, den1 = frac1.split('/')
            num2, den2 = frac2.split('/')
            output = P.initialize.format(frac1, frac2, num1, den1, num2, den2)
            output += P.calc_num
            new_output, num = multiply_integer_integer_integer(num1, num2, 'mul1', 'mul2', 'num')
            output += new_output
            output += P.exit_function1.format(num)
            
            output += P.calc_den
            new_output, den = multiply_integer_integer_integer(den1, den2, 'mul1', 'mul2', 'den')
            output += new_output
            output += P.exit_function2.format(den)
            
            output += P.calc_gcd
            new_output, gcd = gcd_integer_integer_integer(num, den, 'gcd1', 'gcd2')
            output += new_output
            output += P.exit_function3.format(gcd)
            
            output += P.calc_num2
            new_output, num = brief_floordiv_integer_integer_integer(num, gcd, 'div1', 'div2', 'num')
            output += new_output
            output += P.exit_function4.format(num)
            
            output += P.calc_den2
            new_output, den = brief_floordiv_integer_integer_integer(den, gcd, 'div1', 'div2', 'den')
            output += new_output
            output += P.exit_function5.format(den)
            
            result = num + '/' + den
            output += P.calc_result.format(num, den, result)
            output += P.ret.format(result)
            answer = result
            output += P.answer.format(answer)
            assert answer == expected, f'expected: {expected}, got: {answer}'
        
        
        if self.name == 'digit_max_Integer_Integer_Integer':
            num1, num2, expected = re.findall(r'(-?\d+)', q_str)[-3:]
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            # rfft output
            output, answer = digit_max_integer_integer_integer(num1, num2)
            output += P.answer.format(answer)
            assert answer == expected, f'expected: {expected}, got: {answer}'
        

     
        if self.name == 'get_digit_Integer_int_int':
            num1, num2, expected = re.findall(r'(-?\d+)', q_str)[-3:]
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            # rfft output
            output, answer = get_digit_Integer_int_int(num1, num2)
            output += P.answer.format(answer)
            assert answer == expected, f'expected: {expected}, got: {answer}'
            
        if self.name == 'get_digit_Float_int_int':
            num1 = re.findall(r'(-?\d+\.\d+)', q_str)[-1]
            num2, expected = re.findall(r'(-?\d+)', q_str)[-2:]
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            # rfft output
            output, answer = get_digit_Float_int_int(num1, num2)
            output += P.answer.format(answer)
            assert answer == expected, f'expected: {expected}, got: {answer}'
            
        if self.name == 'floordiv_Integer_Integer_Integer':
            num1, num2, expected = re.findall(r'(-?\d+)', q_str)[-3:]
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            input = input.replace('_var1_', 'num1').replace('_var2_', 'num2').replace('_var_res_', 'result')
            # rfft output
            output, answer = floordiv_integer_integer_integer(num1, num2, 'num1', 'num2', 'result')
            # output += P.ret.format(answer)
            output += P.answer.format(answer)
            assert answer == expected, f'expected: {expected}, got: {answer}'
            
        if self.name == 'mod_Integer_Integer_Integer' or self.name == 'mod_easy_Integer_Integer_Integer':
            num1, num2, expected = re.findall(r'(-?\d+)', q_str)[-3:]
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            input = input.replace('_var1_', 'num1').replace('_var2_', 'num2').replace('_var_res_', 'result')
            # rfft output
            output, answer = mod_integer_integer_integer(num1, num2, 'num1', 'num2')
            # output += P.ret.format(answer)
            output += P.answer.format(answer)
            assert answer == expected, f'expected: {expected}, got: {answer}'
        
        if self.name == 'length_Integer_none_int':
            num1, expected = re.findall(r'(-?\d+)', q_str)[-2:]
            input = instruction + rule + "\n\nQ: " + q_str + '\n'
            # rfft output
            output, answer = length_Integer_none_int(num1)
            # output += P.ret.format(answer)
            output += P.answer.format(answer)
            assert str(answer) == expected, f'expected: \'{expected}\', got: \'{answer}\''
        
        return {"input": input,
                    "output": output,
                    "answer": answer}

if __name__ == "__main__":
    pass


'''
add_Integer_Integer_Integer
add_Float_Float_Float
add_Fraction_Fraction_Fraction
add_ScientificNotation_ScientificNotation_ScientificNotation
sub_Integer_Integer_Integer
sub_Float_Float_Float
sub_Fraction_Fraction_Fraction
sub_ScientificNotation_ScientificNotation_ScientificNotation
max_Integer_Integer_Integer
max_Float_Float_Float
max_Fraction_Fraction_Fraction
max_ScientificNotation_ScientificNotation_ScientificNotation
multiply_hard_Integer_Integer_Integer
multiply_hard_Float_Float_Float
multiply_hard_Fraction_Fraction_Fraction
multiply_hard_ScientificNotation_ScientificNotation_ScientificNotation
digit_max_Integer_Integer_Integer
digit_max_Float_Float_Float
digit_add_Integer_Integer_Integer
digit_add_Float_Float_Float
get_digit_Integer_int_int
get_digit_Float_int_int
length_Integer_none_int
length_Float_none_int
truediv_Integer_Integer_Fraction
truediv_Fraction_Fraction_Fraction
floordiv_Integer_Integer_Integer
mod_Integer_Integer_Integer
to_float_Fraction_none_Float
to_float_ScientificNotation_none_Float
to_scient_Integer_none_ScientificNotation
to_scient_Float_none_ScientificNotation
count_Integer_int_int
sig_Integer_int_ScientificNotation








 
    (Task("digit_max", "Float", "Float", "Float"), os.path.join(dataset_path, "Float", "compare")),
    

    (Task("length", "Float", "none", "int"), os.path.join(dataset_path, "Float", "compare")),
    

'''