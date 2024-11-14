class Prompt:
    def __init__(self, dataset_name) -> None:
        self.answer = '''So the answer is {}'''
        if dataset_name == 'find_dot':
            self.rule = '''def find_dot(result_int, pos):
    cnt = 0
    result_dec = ''
    while cnt < pos:
        last_digit = result_int[-1] if result_int else '0'
        result_dec = last_digit + result_dec
        result_int = result_int[:-1]
        cnt += 1
    result_int = result_int.lstrip('0') or '0'
    result_dec = result_dec.rstrip('0') or '0'
    _var_res_ = result_int + '.' + result_dec
    return _var_res_
'''
            self.initialize = '''result_int = '{}'
pos = {}
cnt = 0
result_dec = ''
'''
            self.enter = '''```
while cnt < pos:
```
check the stop criterion
enter the loop
'''
            self.not_enter = '''```
while cnt < pos:
```
check the stop criterion
end the loop
'''
            self.last_digit = '''```
last_digit = result_int[-1] if result_int else '0'
```
last_digit = '{}'
'''
            self.update_result = '''```
result_dec = last_digit + result_dec
result_int = result_int[:-1]
```
result_dec = '{}' + '{}' = '{}'
result_int = '{}'
'''
            self.update_cnt = '''```
cnt += 1
```
cnt = {} + 1 = {}
'''
            self.strip = '''```
result_int = result_int.lstrip('0') or '0'
result_dec = result_dec.rstrip('0') or '0'
```
result_int = '{}'
result_dec = '{}'
'''
            self.merge = '''```
_var_res_ = result_int + '.' + result_dec
```
_var_res_ = '{}' + '.' + '{}' = '{}'
'''
            self.ret = '''```
return _var_res_
```
return '{}'
'''
            
        if dataset_name == "add_Integer_Integer_Integer":
            self.question = 'Add two numbers: {} and {}'
            self.rule = '''
def add(_var1_, _var2_):
    _var_res_ = ''
    carry = 0
    # Main Loop
    while _var1_ or _var2_:
        digit1 = int(_var1_[-3:]) if _var1_ else 0
        digit2 = int(_var2_[-3:]) if _var2_ else 0
        total = digit1 + digit2 + carry
        _var_res_ = str(total%1000) + _var_res_
        carry = total//1000
        _var1_ = _var1_[:-3] if _var1_ else _var1_
        _var2_ = _var2_[:-3] if _var2_ else _var2_
    if carry:
        _var_res_ = str(carry) + _var_res_
    _var_res_ = _var_res_.lstrip('0') or '0'
    return _var_res_'''
            self.initialize = '''_var1_ = '{}'
_var2_ = '{}'
_var_res_ = ''
carry = 0
'''
            self.last_digit = '''```
digit1 = int(_var1_[-3:]) if _var1_ else 0
digit2 = int(_var2_[-3:]) if _var2_ else 0
```
_var1_ = '{}'
_var2_ = '{}'
digit1 = {}
digit2 = {}
'''
            self.sum = '''```
total = digit1 + digit2 + carry
```
carry = {}
total = {} + {} + {} = {}
'''
            self.update_result = '''```
_var_res_ = str(total%1000) + _var_res_
carry = total//1000         
```
total % 1000 = {}%1000 = {}
_var_res_ = '{}' + '{}' = '{}'
carry = {}//1000 = {}
'''
            self.enter = '''```
while _var1_ or _var2_:
```
check the stop criterion
enter the loop
'''
            self.not_enter = '''```
while _var1_ or _var2_
```
check the stop criterion
_var1_ = '' and _var2_ = ''
end the loop
'''
            self.update_nums = '''```
_var1_ = _var1_[:-3]
_var2_ = _var2_[:-3]
```
_var1_ = '{}'
_var2_ = '{}'
'''
            self.carry_true = '''```
if carry:
    _var_res_ = str(carry) + _var_res_
```
carry = {}
_var_res_ = '{}' + '{}' = '{}'
''' 
            self.carry_false = '''```
if carry:
    _var_res_ = str(carry) + _var_res_
```
carry = {}
pass
''' 
            self.lstrip = '''```
_var_res_ = _var_res_.lstrip('0') or '0'
```
_var_res_ = '{}'
'''
            self.ret = '''```
return _var_res_
```
return '{}'
'''
        
        if dataset_name == "add_Float_Float_Float":
            self.question = 'Add two numbers: {} and {}'
            self.rule = '''
def add_float(_var1_, _var2_):
    int1, dec1 = _var1_.split('.')
    int2, dec2 = _var2_.split('.')
    len1 = len(dec1)
    len2 = len(dec2)
    if len1 < len2:
        while len1 < len2:
            dec1 += '0'
            len1 += 1
    elif len1 > len2:
        while len1 > len2:
            dec2 += '0'
            len2 += 1
    full1 = int1 + dec1
    full2 = int2 + dec2
    result = add_integer(full1, full2)
    _var_res_ = find_dot(result, len1)
    return _var_res_
    '''
            self.initialize = '''_var1_ = '{}'
_var2_ = '{}'
```
int1, dec1 = _var1_.split('.')
int2, dec2 = _var2_.split('.')
```
int1 = '{}'
dec1 = '{}'
int2 = '{}'
dec2 = '{}'
```
len1 = len(dec1)
len2 = len(dec2)
```
len1 = {}
len2 = {}
'''
            self.condition1 = '''check the condition
len1 = {} {} len2 = {}
'''
            self.not_branch = '''do not enter the branch
'''
            self.if1 = '''enter the first branch
'''
            self.enter1 = '''```
while len1 < len2:
```
enter the loop
'''
            self.update_vars1 = '''```
dec1 += '0'
len1 += 1
```
dec1 = '{}' + '0' = '{}'
len1 = {} + 1 = {}
'''
            self.out1 = '''```
while len1 < len2:
```
check the stop criterion
end the loop
'''
            self.if2 = '''enter the second branch
'''
            self.enter2 = '''```
while len1 > len2:
```
enter the loop
'''
            self.update_vars2 = '''```
dec2 += '0'
len2 += 1
```
dec2 = '{}' + '0' = '{}'
len2 = {} + 1 = {}
'''
            self.out2 = '''```
while len1 > len2:
```
check the stop criterion
end the loop
'''
            self.full = '''```
full1 = int1 + dec1
full2 = int2 + dec2
```
full1 = '{}' + '{}' = '{}'
full2 = '{}' + '{}' = '{}'
'''
            self.add_integer = '''```
result = add_integer(full1, full2)
```
enter function: add_integer
'''
            self.exit_function = '''exit function: add_integer
result = '{}'
'''
            self.find_dot = '''```
_var_res_ = find_dot(result, len1)
```
enter function: find_dot
'''
            self.exit_function2 = '''exit function: find_dot
_var_res_ = '{}'
'''
            self.ret = '''```
return _var_res_
```
return '{}'
'''
        
        if dataset_name == "add_Fraction_Fraction_Fraction" or dataset_name == "add_easy_Fraction_Fraction_Fraction":
            self.question = 'Add two fractions: {} and {}'
            self.rule = '''
def add_fraction(frac1, frac2):
    num1, den1 = frac1.split('/')
    num2, den2 = frac2.split('/')
    den = multiply_integer(den1, den2)
    tmp1 = multiply_integer(num1, den2)
    tmp2 = multiply_integer(num2, den1)
    num = add_integer(tmp1, tmp2)
    gcd_num_den = gcd(num, den)
    num = floordiv(num, gcd_num_den)
    den = floordiv(den, gcd_num_den)
    result = num + '/' + den
    return result'''
            self.initialize = '''frac1 = '{}'
frac2 = '{}'
```
num1, den1 = frac1.split('/')
num2, den2 = frac2.split('/')
```
num1 = '{}'
den1 = '{}'
num2 = '{}'
den2 = '{}'
'''
            self.calc_den = '''```
den = multiply_integer(den1, den2)
```
enter function: multiply_integer
'''
            self.exit_function1 = '''exit function: multiply_integer
den = '{}'
'''
            self.calc_tmp1 = '''```
tmp1 = multiply_integer(num1, den2)
```
enter function: multiply_integer
'''
            self.exit_function2 = '''exit function: multiply_integer
tmp1 = '{}'
'''
            self.calc_tmp2 = '''```
tmp2 = multiply_integer(num2, den1)
```
enter function: multiply_integer
'''
            self.exit_function3 = '''exit function: multiply_integer
tmp2 = '{}'
'''
            self.add_num = '''```
num = add_integer(tmp1, tmp2)
```
enter function: add_integer
'''
            self.exit_function4 = '''exit function: add_integer
num = '{}'
'''
            self.calc_gcd = '''```
gcd_num_den = gcd(num, den)
```
enter function: gcd
'''
            self.exit_function5 = '''exit function: gcd
gcd_num_den = '{}'
'''
            self.calc_num2 = '''```
num = floordiv(num, gcd_num_den)
```
enter function: floordiv
'''
            self.exit_function6 = '''exit function: floordiv
num = '{}'
'''
            self.calc_den2 = '''```
den = floordiv(den, gcd_num_den)
```
enter function: floordiv
'''
            self.exit_function7 = '''exit function: floordiv
den = '{}'
'''
            self.calc_result = '''```
result = num + '/' + den
```
result = '{}' + '/' + '{}' = '{}'
'''
        
        if dataset_name == "add_ScientificNotation_ScientificNotation_ScientificNotation":
            self.question = 'Add two numbers: {} and {}'
            self.rule = '''
def add_ScientificNotation(num1, num2):
    base1, exp1 = num1.split('e')
    base2, exp2 = num2.split('e')
    exp1, exp2 = int(exp1), int(exp2)
    if exp1 < exp2:
        base1, base2 = base2, base1
        exp1, exp2 = exp2, exp1
        
    int1, dec1 = base1.split('.')
    cnt = exp1 - exp2
    while cnt > 0:
        int1 += dec1[0] if dec1 else '0'
        dec1 = dec1[1:] if dec1 else dec1
        cnt -= 1
        
    if dec1 == '':
        dec1 = '0'
    base1 = int1 + '.' + dec1
    result_exp = exp2
    result_base = add_float(base1, base2)
    int_part, dec_part = result_base.split('.')
    while len(int_part) > 1:
        dec_part = int_part[-1] + dec_part
        int_part = int_part[:-1]
        result_exp += 1
    dec_part = dec_part.rstrip('0') or '0'
    result_base = int_part + '.' + dec_part
    result = result_base + 'e' + str(result_exp)
    return result
'''
            self.initialize = '''num1 = '{}'
num2 = '{}'
'''
            self.split = '''```
base1, exp1 = num1.split('e')
base2, exp2 = num2.split('e')
exp1, exp2 = int(exp1), int(exp2)
```
base1 = '{}'
exp1 = {}
base2 = '{}'
exp2 = {}
'''
            self.condition = '''```
if exp1 < exp2:
```
check the condition
exp1 = {} {} exp2 = {}
'''
            self.not_branch = '''do not enter the branch
'''
            self.if1 = '''enter the branch
```
base1, base2 = base2, base1
exp1, exp2 = exp2, exp1
```
base1 = '{}'
base2 = '{}'
exp1 = {}
exp2 = {}
'''
            self.split2 = '''```
int1, dec1 = base1.split('.')
```
int1 = '{}'
dec1 = '{}'
```
cnt = exp1 - exp2
```
cnt = {} - {} = {}
'''
            self.enter = '''```
while cnt > 0:
```
enter the loop
'''
            self.update_nums = '''```
int1 += dec1[0] if dec1 else '0'
dec1 = dec1[1:] if dec1 else dec1
cnt -= 1
```
int1 = '{}' + '{}' = '{}'
dec1 = '{}'
cnt = {} - 1 = {}
'''
            self.out = '''```
while cnt > 0:
```
check the stop criterion
end the loop
'''
            self.update_dec = '''```
if dec1 == '':
    dec1 = '0'
```
dec1 = '{}'
'''
            self.update_base1 = '''```
base1 = int1 + '.' + dec1
```
base1 = '{}' + '.' + '{}' = '{}'
'''
            self.update_exp = '''```
result_exp = exp2
```
result_exp = '{}'
'''
            self.add_float = '''```
result_base = add_float(base1, base2)
```
enter function: add_float
'''
            self.exit_function1 = '''exit function: add_float
result_base = '{}'
'''
            self.split3 = '''```
int_part, dec_part = result_base.split('.')
```
int_part = '{}'
dec_part = '{}'
'''
            self.enter2 = '''```
while len(int_part) > 1:
```
enter the loop
'''
            self.update_parts = '''```
dec_part = int_part[-1] + dec_part
int_part = int_part[:-1]
```
dec_part = '{}' + '{}' = '{}'
int_part = '{}'
'''
            self.update_exp2 = '''```
result_exp += 1
```
result_exp = {} + 1 = {}
'''
            self.out2 = '''```
while len(int_part) > 1:
```
check the stop criterion
end the loop
'''
            self.rstrip = '''```
dec_part = dec_part.rstrip('0') or '0'
```
dec_part = '{}'
'''
            self.update_result_base = '''```
result_base = int_part + '.' + dec_part
```
result_base = '{}' + '.' + '{}' = '{}'
'''
            self.merge = '''```
result = result_base + 'e' + str(result_exp)
```
result = '{}' + 'e' + '{}' = '{}'
'''
            self.ret = '''```
return result
```
return '{}'
'''


        if dataset_name == "sub_Integer_Integer_Integer":
            self.question = 'Add two numbers: {} and {}'
            self.rule = '''
def sub_integer(num1, num2):
    result = ''
    borrow = 0
    # Main Loop
    while num1 or num2 or borrow:
        digit1 = int(num1[-3:]) if num1 else 0
        digit2 = int(num2[-3:]) if num2 else 0
        total = digit1 - digit2 - borrow
        if total < 0:
            total += 1000
            borrow = 1
        else:
            borrow = 0
        result = str(total) + result
        num1 = num1[:-3] if num1 else num1
        num2 = num2[:-3] if num2 else num2
    
    result = result.lstrip('0') or '0'
    return result'''
            self.initialize = '''num1 = '{}'
num2 = '{}'
result = ''
borrow = 0
'''
            self.last_digit = '''```
digit1 = int(num1[-1]) if num1 else 0
digit2 = int(num2[-1]) if num2 else 0
```
num1 = {}
num2 = {}
digit1 = {}
digit2 = {}
'''
            self.sub = '''```
total = digit1 - digit2 - borrow
```
borrow = {}
total = {} - {} - {} = {}
'''
            self.borrow_true = '''```
if total < 0:
    total += 10
    borrow = 1
else:
    borrow = 0
```
total = {} < 0
enter if
total = {} + 10 = {}
borrow = 1
'''
            self.borrow_false = '''```
if total < 0:
    total += 10
    borrow = 1
else:
    borrow = 0
```
total = {} >= 0
enter else
borrow = 0
'''
            self.update_result = '''```
result = str(total) + result       
```
result = '{}' + '{}' = '{}'
'''
            self.enter = '''```
while num1 or num2 or borrow:
```
check the stop criterion
enter the loop
'''
            self.not_enter = '''```
while num1 or num2 or borrow:
```
check the stop criterion
num1 = '' and num2 = ''
end the loop
'''
            self.update_nums = '''```
num1 = num1[:-1]
num2 = num2[:-1]
```
num1 = '{}'
num2 = '{}'
'''
            self.lstrip = '''```
result = result.lstrip('0') or '0'
```
result = '{}'
''' 
            self.ret = '''```
return result
```
return '{}'
'''
        
        if dataset_name == 'sub_Float_Float_Float':
            self.question = 'Subtract two numbers: {} and {}'
            self.rule = '''
def sub_float(_var1_, _var2_):
    int1, dec1 = _var1_.split('.')
    int2, dec2 = _var2_.split('.')
    len1 = len(dec1)
    len2 = len(dec2)
    if len1 < len2:
        while len1 < len2:
            dec1 += '0'
            len1 += 1
    elif len1 > len2:
        while len1 > len2:
            dec2 += '0'
            len2 += 1
    full1 = int1 + dec1
    full2 = int2 + dec2
    result = sub_integer(full1, full2)
    _var_res_ = find_dot(result, len1)
    return _var_res_
'''
            self.initialize = '''_var1_ = '{}'
_var2_ = '{}'
```
int1, dec1 = _var1_.split('.')
int2, dec2 = _var2_.split('.')
```
int1 = '{}'
dec1 = '{}'
int2 = '{}'
dec2 = '{}'
```
len1 = len(dec1)
len2 = len(dec2)
```
len1 = {}
len2 = {}
'''
            self.condition1 = '''check the condition
len1 = {} {} len2 = {}
'''
            self.not_branch = '''do not enter the branch
'''
            self.if1 = '''enter the first branch
'''
            self.enter1 = '''```
while len1 < len2:
```
enter the loop
'''
            self.update_vars1 = '''```
dec1 += '0'
len1 += 1
```
dec1 = '{}' + '0' = '{}'
len1 = {} + 1 = {}
'''
            self.out1 = '''```
while len1 < len2:
```
check the stop criterion
end the loop
'''
            self.if2 = '''enter the second branch
'''
            self.enter2 = '''```
while len1 > len2:
```
enter the loop
'''
            self.update_vars2 = '''```
dec2 += '0'
len2 += 1
```
dec2 = '{}' + '0' = '{}'
len2 = {} + 1 = {}
'''
            self.out2 = '''```
while len1 > len2:
```
check the stop criterion
end the loop
'''
            self.full = '''```
full1 = int1 + dec1
full2 = int2 + dec2
```
full1 = '{}' + '{}' = '{}'
full2 = '{}' + '{}' = '{}'
'''
            self.sub_integer = '''```
result = sub_integer(full1, full2)
```
enter function: sub_integer
'''
            self.exit_function = '''exit function: sub_integer
result = '{}'
'''
            self.find_dot = '''```
_var_res_ = find_dot(result, len1)
```
enter function: find_dot
'''
            self.exit_function2 = '''exit function: find_dot
_var_res_ = '{}'
'''
            self.ret = '''```
return _var_res_
```
return '{}'
'''
        
        if dataset_name == 'sub_Fraction_Fraction_Fraction':
            self.question = 'Subtract two fractions: {} and {}'
            self.rule = '''
def sub_fraction(num1, num2):
    num1, den1 = num1.split('/')
    num2, den2 = num2.split('/')
    den = multiply_integer(den1, den2)
    tmp1 = multiply_integer(num1, den2)
    tmp2 = multiply_integer(num2, den1)
    num = sub_integer(tmp1, tmp2)
    gcd_num_den = gcd(num, den)
    num = floordiv(num, gcd_num_den)
    den = floordiv(den, gcd_num_den)
    result = num + '/' + den
    return result
'''
            self.initialize = '''frac1 = '{}'
frac2 = '{}'
```
num1, den1 = frac1.split('/')
num2, den2 = frac2.split('/')
```
num1 = '{}'
den1 = '{}'
num2 = '{}'
den2 = '{}'
'''
            self.calc_den = '''```
den = multiply_integer(den1, den2)
```
enter function: multiply_integer
'''
            self.exit_function1 = '''exit function: multiply_integer
den = '{}'
'''
            self.calc_tmp1 = '''```
tmp1 = multiply_integer(num1, den2)
```
enter function: multiply_integer
'''
            self.exit_function2 = '''exit function: multiply_integer
tmp1 = '{}'
'''
            self.calc_tmp2 = '''```
tmp2 = multiply_integer(num2, den1)
```
enter function: multiply_integer
'''
            self.exit_function3 = '''exit function: multiply_integer
tmp2 = '{}'
'''
            self.sub_num = '''```
num = sub_integer(tmp1, tmp2)
```
enter function: sub_integer
'''
            self.exit_function4 = '''exit function: sub_integer
num = '{}'
'''
            self.calc_gcd = '''```
gcd_num_den = gcd(num, den)
```
enter function: gcd
'''
            self.exit_function5 = '''exit function: gcd
gcd_num_den = '{}'
'''
            self.calc_num2 = '''```
num = floordiv(num, gcd_num_den)
```
enter function: floordiv
'''
            self.exit_function6 = '''exit function: floordiv
num = '{}'
'''
            self.calc_den2 = '''```
den = floordiv(den, gcd_num_den)
```
enter function: floordiv
'''
            self.exit_function7 = '''exit function: floordiv
den = '{}'
'''
            self.calc_result = '''```
result = num + '/' + den
```
result = '{}' + '/' + '{}' = '{}'
'''
        
        if dataset_name == 'sub_ScientificNotation_ScientificNotation_ScientificNotation':
            self.question = 'Subtract two numbers: {} and {}'
            self.rule = '''
def sub_ScientificNotation(num1, num2):
    base1, exp1 = num1.split('e')
    base2, exp2 = num2.split('e')
    exp1, exp2 = int(exp1), int(exp2)
    int1, dec1 = base1.split('.')
    cnt = exp1 - exp2
    while cnt > 0:
        int1 += dec1[0] if dec1 else '0'
        dec1 = dec1[1:] if dec1 else dec1
        cnt -= 1
        
    if dec1 == '':
        dec1 = '0'
    base1 = int1 + '.' + dec1
    result_exp = exp2
    result_base = sub_float(base1, base2)
    int_part, dec_part = result_base.split('.')
    while len(int_part) > 1:
        dec_part = int_part[-1] + dec_part
        int_part = int_part[:-1]
        result_exp += 1
    while int_part == '0':
        int_part = dec_part[0]
        dec_part = dec_part[1:]
        result_exp -= 1
    dec_part = dec_part.rstrip('0') or '0'
    result_base = int_part + '.' + dec_part
    result = result_base + 'e' + str(result_exp)
    return result
'''
            self.initialize = '''num1 = '{}'
num2 = '{}'
'''
            self.split = '''```
base1, exp1 = num1.split('e')
base2, exp2 = num2.split('e')
exp1, exp2 = int(exp1), int(exp2)
```
base1 = '{}'
exp1 = {}
base2 = '{}'
exp2 = {}
'''
            self.split2 = '''```
int1, dec1 = base1.split('.')
```
int1 = '{}'
dec1 = '{}'
```
cnt = exp1 - exp2
```
cnt = {} - {} = {}
'''
            self.enter = '''```
while cnt > 0:
```
enter the loop
'''
            self.update_nums = '''```
int1 += dec1[0] if dec1 else '0'
dec1 = dec1[1:] if dec1 else dec1
cnt -= 1
```
int1 = '{}' + '{}' = '{}'
dec1 = '{}'
cnt = {} - 1 = {}
'''
            self.out = '''```
while cnt > 0:
```
check the stop criterion
end the loop
'''
            self.update_dec = '''```
if dec1 == '':
    dec1 = '0'
```
dec1 = '{}'
'''
            self.update_base1 = '''```
base1 = int1 + '.' + dec1
```
base1 = '{}' + '.' + '{}' = '{}'
'''
            self.update_exp = '''```
result_exp = exp2
```
result_exp = '{}'
'''
            self.sub_float = '''```
result_base = sub_float(base1, base2)
```
enter function: sub_float
'''
            self.exit_function1 = '''exit function: sub_float
result_base = '{}'
'''
            self.split3 = '''```
int_part, dec_part = result_base.split('.')
```
int_part = '{}'
dec_part = '{}'
'''
            self.enter2 = '''```
while len(int_part) > 1:
```
enter the loop
'''
            self.update_parts = '''```
dec_part = int_part[-1] + dec_part
int_part = int_part[:-1]
```
dec_part = '{}' + '{}' = '{}'
int_part = '{}'
'''
            self.update_exp2 = '''```
result_exp += 1
```
result_exp = {} + 1 = {}
'''
            self.out2 = '''```
while len(int_part) > 1:
```
check the stop criterion
end the loop
'''
            self.enter3 = '''```
while int_part == '0':
```
enter the loop
'''
            self.update_parts2 = '''```
int_part = dec_part[0]
dec_part = dec_part[1:]
```
int_part = '{}'
dec_part = '{}'
'''
            self.update_exp3 = '''```
result_exp -= 1
```
result_exp = {} - 1 = {}
'''
            self.out3 = '''```
while int_part == '0':
```
check the stop criterion
end the loop
'''
            self.rstrip = '''```
dec_part = dec_part.rstrip('0') or '0'
```
dec_part = '{}'
'''
            self.update_result_base = '''```
result_base = int_part + '.' + dec_part
```
result_base = '{}' + '.' + '{}' = '{}'
'''
            self.merge = '''```
result = result_base + 'e' + str(result_exp)
```
result = '{}' + 'e' + '{}' = '{}'
'''
            self.ret = '''```
return result
```
return '{}'
'''

        if dataset_name == "max_Integer_Integer_Integer" or dataset_name == "max_hard_Integer_Integer_Integer":
            self.question = 'Get the maximal number: {} and {}'
            self.rule = '''
def max_int(num1, num2):
    len1 = len(num1)
    len2 = len(num2)
    if len1 < len2:
        return num2
    elif len1 > len2:
        return num1
    elif len1 == len2:
        num1_copy = num1
        num2_copy = num2
        while num1 and num2:
            digit1 = num1[0]
            digit2 = num2[0]
            if digit1 < digit2:
                return num2_copy
            elif digit1 > digit2:
                return num1_copy
            num1 = num1[1:]
            num2 = num2[1:]
    return num1_copy
'''
            self.initialize = '''num1 = '{}'
num2 = '{}'
```
len1 = len(num1)
len2 = len(num2)
```
len1 = {}
len2 = {}
'''
            self.condition1 = '''check the condition
len1 = {} {} len2 = {}
'''
            self.if1 = '''enter the first branch
```
return num2
```
return '{}'
'''
            self.if2 = '''enter the second branch
```
return num1
```
return = '{}'
'''
            self.if3 = '''enter the third branch
```
num1_copy = num1
num2_copy = num2
```
num1_copy = '{}'
num2_copy = '{}'
'''
            self.enter = '''```
while num1 and num2:
```
check the stop criterion
enter the loop
'''
            self.not_enter = '''```
while num1 and num2:
```
check the stop criterion
end the loop
'''
            self.first_digit = '''```
digit1 = num1[0]
digit2 = num2[0]
```
digit1 = '{}'
digit2 = '{}'
'''
            self.compare = '''check the condition
digit1 = {} {} digit2 = {}
'''
            self.if4 = '''enter the first branch
```
return num2_copy
```
return = '{}'
'''
            self.if5 = '''enter the second branch
```
return num1_copy
```
return = '{}'
'''
            self.update_nums = '''do not enter the branches
```
num1 = num1[1:]
num2 = num2[1:]
```
num1 = '{}'
num2 = '{}'
'''
            self.ret = '''```
return num1_copy
```
return '{}'
'''
        
        if dataset_name == 'max_Float_Float_Float' or dataset_name == 'max_hard_Float_Float_Float':
            self.question = 'Get the maximal number: {} and {}'
            self.rule = '''
def max_float(num1, num2):
    int1, dec1 = num1.split('.')
    int2, dec2 = num2.split('.')
    len1 = len(dec1)
    len2 = len(dec2)
    if len1 < len2:
        while len1 < len2:
            dec1 += '0'
            len1 += 1
    elif len1 > len2:
        while len1 > len2:
            dec2 += '0'
            len2 += 1
    full1 = int1 + dec1
    full2 = int2 + dec2
    len1 = len(full1)
    len2 = len(full2)
    if len1 < len2:
        return num2
    elif len1 > len2:
        return num1
    elif len1 == len2:
        num1_copy = num1
        num2_copy = num2
        while full1 or full2:
            digit1 = full1[0]
            digit2 = full2[0]
            if digit1 < digit2:
                return num2_copy
            elif digit1 > digit2:
                return num1_copy
            full1 = full1[1:]
            full2 = full2[1:]
    return num1_copy
'''
            self.initialize = '''num1 = '{}'
num2 = '{}'
```
int1, dec1 = num1.split('.')
int2, dec2 = num2.split('.')
```
int1 = '{}'
dec1 = '{}'
int2 = '{}'
dec2 = '{}'
```
len1 = len(dec1)
len2 = len(dec2)
```
len1 = {}
len2 = {}
'''
            self.pcondition1 = '''check the condition
len1 = {} {} len2 = {}
'''
            self.pnot_branch = '''do not enter the branch
'''
            self.pif1 = '''enter the first branch
'''
            self.penter1 = '''```
while len1 < len2:
```
enter the loop
'''
            self.pupdate_vars1 = '''```
dec1 += '0'
len1 += 1
```
dec1 = '{}' + '0' = '{}'
len1 = {} + 1 = {}
'''
            self.pout1 = '''```
while len1 < len2:
```
check the stop criterion
end the loop
'''
            self.pif2 = '''enter the second branch
'''
            self.penter2 = '''```
while len1 > len2:
```
enter the loop
'''
            self.pupdate_vars2 = '''```
dec2 += '0'
len2 += 1
```
dec2 = '{}' + '0' = '{}'
len2 = {} + 1 = {}
'''
            self.pout2 = '''```
while len1 > len2:
```
check the stop criterion
end the loop
'''
            self.pfull = '''```
full1 = int1 + dec1
full2 = int2 + dec2
```
full1 = '{}' + '{}' = '{}'
full2 = '{}' + '{}' = '{}'
```
len1 = len(full1)
len2 = len(full2)
```
len1 = {}
len2 = {}
'''
            self.condition1 = '''check the condition
len1 = {} {} len2 = {}
'''
            self.if1 = '''enter the first branch
```
return num2
```
return '{}'
'''
            self.if2 = '''enter the second branch
```
return num1
```
return = '{}'
'''
            self.if3 = '''enter the third branch
```
num1_copy = num1
num2_copy = num2
```
num1_copy = '{}'
num2_copy = '{}'
'''
            self.enter = '''```
while full1 and full2:
```
check the stop criterion
enter the loop
'''
            self.not_enter = '''```
while full1 and full2:
```
check the stop criterion
end the loop
'''
            self.first_digit = '''```
digit1 = full1[0]
digit2 = full2[0]
```
digit1 = '{}'
digit2 = '{}'
'''
            self.compare = '''check the condition
digit1 = {} {} digit2 = {}
'''
            self.if4 = '''enter the first branch
```
return num2_copy
```
return = '{}'
'''
            self.if5 = '''enter the second branch
```
return num1_copy
```
return = '{}'
'''
            self.update_nums = '''do not enter the branches
```
full1 = full1[1:]
full2 = full2[1:]
```
full1 = '{}'
full2 = '{}'
'''
            self.ret = '''```
return num1_copy
```
return '{}'
'''
        
        if dataset_name == "max_Fraction_Fraction_Fraction":
            self.question = 'Get the maximal number: {} and {}'
            self.rule = '''
def max_fraction(frac1, frac2):
    num1, den1 = frac1.split('/')
    num2, den2 = frac2.split('/')
    tmp1 = multiply_integer(num1, den2)
    tmp2 = multiply_integer(num2, den1)
    
    len1 = len(tmp1)
    len2 = len(tmp2)
    if len1 < len2:
        return frac2
    elif len1 > len2:
        return frac1
    elif len1 == len2:
        while tmp1 or tmp2:
            digit1 = tmp1[0]
            digit2 = tmp2[0]
            if digit1 < digit2:
                return frac2
            elif digit1 > digit2:
                return frac1
            tmp1 = tmp1[1:]
            tmp2 = tmp2[1:]
    return frac1
'''
            self.initialize = '''frac1 = '{}'
frac2 = '{}'
```
num1, den1 = frac1.split('/')
num2, den2 = frac2.split('/')
```
num1 = '{}'
den1 = '{}'
num2 = '{}'
den2 = '{}'
'''
            self.calc_tmp1 = '''```
tmp1 = multiply_integer(num1, den2)
```
enter function: multiply_integer
'''
            self.exit_function2 = '''exit function: multiply_integer
tmp1 = {}
'''
            self.calc_tmp2 = '''```
tmp2 = multiply_integer(num2, den1)
```
enter function: multiply_integer
'''
            self.exit_function3 = '''exit function: multiply_integer
tmp2 = {}
'''
            self.calc_len = '''```
len1 = len(tmp1)
len2 = len(tmp2)
```
len1 = {}
len2 = {}
'''
            self.condition1 = '''check the condition
len1 = {} {} len2 = {}
'''
            self.if1 = '''enter the first branch
```
return frac2
```
return '{}'
'''
            self.if2 = '''enter the second branch
```
return frac1
```
return = '{}'
'''
            self.if3 = '''enter the third branch
'''
            self.enter = '''```
while tmp1 and tmp2:
```
check the stop criterion
enter the loop
'''
            self.not_enter = '''```
while tmp1 and tmp2:
```
check the stop criterion
end the loop
'''
            self.first_digit = '''```
digit1 = tmp1[0]
digit2 = tmp2[0]
```
digit1 = '{}'
digit2 = '{}'
'''
            self.compare = '''check the condition
digit1 = {} {} digit2 = {}
'''
            self.if4 = '''enter the first branch
```
return frac2
```
return = '{}'
'''
            self.if5 = '''enter the second branch
```
return frac1
```
return = '{}'
'''
            self.update_nums = '''do not enter the branches
```
tmp1 = tmp1[1:]
tmp2 = tmp2[1:]
```
tmp1 = '{}'
tmp2 = '{}'
'''
            self.ret = '''```
return frac1
```
return '{}'
'''
        
        if dataset_name == 'max_ScientificNotation_ScientificNotation_ScientificNotation' or dataset_name == 'max_hard_ScientificNotation_ScientificNotation_ScientificNotation':
            self.question = 'Get the maximal number: {} and {}'
            self.rule = '''
def max_ScientificNotation(num1, num2):
    base1, exp1 = num1.split('e')
    base2, exp2 = num2.split('e')
    exp1, exp2 = int(exp1), int(exp2)
    if exp1 < exp2:
        return num2
    elif exp1 > exp2:
        return num1
        
    # compare base1 and base2
    num1_copy = num1
    num2_copy = num2
    num1 = base1
    num2 = base2
    int1, dec1 = num1.split('.')
    int2, dec2 = num2.split('.')
    len1 = len(dec1)
    len2 = len(dec2)
    if len1 < len2:
        while len1 < len2:
            dec1 += '0'
            len1 += 1
    elif len1 > len2:
        while len1 > len2:
            dec2 += '0'
            len2 += 1
    full1 = int1 + dec1
    full2 = int2 + dec2
    len1 = len(full1)
    len2 = len(full2)
    if len1 < len2:
        return num2_copy
    elif len1 > len2:
        return num1_copy
    elif len1 == len2:
        while full1 or full2:
            digit1 = full1[0]
            digit2 = full2[0]
            if digit1 < digit2:
                return num2_copy
            elif digit1 > digit2:
                return num1_copy
            full1 = full1[1:]
            full2 = full2[1:]
    return num1_copy
'''
            self.sinitialize = '''num1 = '{}'
num2 = '{}'
'''
            self.ssplit = '''```
base1, exp1 = num1.split('e')
base2, exp2 = num2.split('e')
exp1, exp2 = int(exp1), int(exp2)
```
base1 = '{}'
exp1 = {}
base2 = '{}'
exp2 = {}
'''
            self.scondition = '''```
if exp1 < exp2:
```
check the condition
exp1 = {} {} exp2 = {}
'''
            self.snot_branch = '''do not enter the branches
'''
            self.sif1 = '''enter the first branch
```
return num2
```
return '{}'
'''
            self.sif2 = '''enter the second branch
```
return num1
```
return = '{}'
'''
            self.initialize = '''```
num1_copy = num1
num2_copy = num2
num1 = base1
num2 = base2
```
num1_copy = '{}'
num2_copy = '{}'        
num1 = '{}'
num2 = '{}'
```
int1, dec1 = num1.split('.')
int2, dec2 = num2.split('.')
```
int1 = '{}'
dec1 = '{}'
int2 = '{}'
dec2 = '{}'
```
len1 = len(dec1)
len2 = len(dec2)
```
len1 = {}
len2 = {}
'''
            self.pcondition1 = '''check the condition
len1 = {} {} len2 = {}
'''
            self.pnot_branch = '''do not enter the branch
'''
            self.pif1 = '''enter the first branch
'''
            self.penter1 = '''```
while len1 < len2:
```
enter the loop
'''
            self.pupdate_vars1 = '''```
dec1 += '0'
len1 += 1
```
dec1 = '{}' + '0' = '{}'
len1 = {} + 1 = {}
'''
            self.pout1 = '''```
while len1 < len2:
```
check the stop criterion
end the loop
'''
            self.pif2 = '''enter the second branch
'''
            self.penter2 = '''```
while len1 > len2:
```
enter the loop
'''
            self.pupdate_vars2 = '''```
dec2 += '0'
len2 += 1
```
dec2 = '{}' + '0' = '{}'
len2 = {} + 1 = {}
'''
            self.pout2 = '''```
while len1 > len2:
```
check the stop criterion
end the loop
'''
            self.pfull = '''```
full1 = int1 + dec1
full2 = int2 + dec2
```
full1 = '{}' + '{}' = '{}'
full2 = '{}' + '{}' = '{}'
```
len1 = len(full1)
len2 = len(full2)
```
len1 = {}
len2 = {}
'''
            self.condition1 = '''check the condition
len1 = {} {} len2 = {}
'''
            self.if1 = '''enter the first branch
```
return num2_copy
```
return '{}'
'''
            self.if2 = '''enter the second branch
```
return num1_copy
```
return = '{}'
'''
            self.if3 = '''enter the third branch
'''
            self.enter = '''```
while full1 and full2:
```
check the stop criterion
enter the loop
'''
            self.not_enter = '''```
while full1 and full2:
```
check the stop criterion
end the loop
'''
            self.first_digit = '''```
digit1 = full1[0]
digit2 = full2[0]
```
digit1 = '{}'
digit2 = '{}'
'''
            self.compare = '''check the condition
digit1 = {} {} digit2 = {}
'''
            self.if4 = '''enter the first branch
```
return num2_copy
```
return = '{}'
'''
            self.if5 = '''enter the second branch
```
return num1_copy
```
return = '{}'
'''
            self.update_nums = '''do not enter the branches
```
full1 = full1[1:]
full2 = full2[1:]
```
full1 = '{}'
full2 = '{}'
'''
            self.ret = '''```
return num1_copy
```
return '{}'
'''

        
        if dataset_name == "multiply_hard_Integer_Integer_Integer" or dataset_name == 'multiply_easy_Integer_Integer_Integer':
            self.question = 'Multiply two numbers: {} and {}'
            self.rule = '''
def multiply_integer(num1, num2):
    result = 0
    base = 0
    while num1:
        digit1 = int(num1[-1])
        # multiply num2 with digit1
        temp = int(num2) * digit1
        temp *= 10**base
        base += 1
        
        # add temp to result
        result = result + temp
        num1 = num1[:-1]
    return result'''
            self.initialize = '''num1 = '{}'
num2 = '{}'
result = 0
base = 0
'''
            self.enter = '''```
while num1:
```
check the stop criterion
enter the loop
'''
            self.not_enter = '''```
while num1:
```
check the stop criterion
end the loop
'''
            self.last_digit = '''```
digit1 = int(num1[-1])
```
digit1 = {}
'''
            self.multiply = '''```
temp = int(num2) * digit1
```
temp = {} * {} = {}
'''
            self.update_temp = '''```
temp *= 10**base
base += 1
```
temp = {} * 10**{} = {}
base = {} + 1 = {}
'''
            
            self.add_temp = '''```
result = result + temp
```
result = {} + {} = {}
'''
            self.update_result_num = '''```
num1 = num1[:-1]
```
num1 = '{}'
'''
            self.ret = '''```
return result
```
return {}
'''
        
        if dataset_name == "multiply_hard_Float_Float_Float" or dataset_name == 'multiply_easy_Float_Float_Float':
            self.question = 'Multiply two numbers: {} and {}'
            self.rule = '''
def multiply_float(_var1_, _var2_):
    int1, dec1 = _var1_.split('.')
    int2, dec2 = _var2_.split('.')
    len1 = len(dec1)
    len2 = len(dec2)
    full1 = int1 + dec1
    full2 = int2 + dec2
    result = multiply_integer(full1, full2)
    _var_res_ = find_dot(result, len1+len2)
    return _var_res_
'''
            self.initialize = '''_var1_ = '{}'
_var2_ = '{}'
```
int1, dec1 = _var1_.split('.')
int2, dec2 = _var2_.split('.')
```
int1 = '{}'
dec1 = '{}'
int2 = '{}'
dec2 = '{}'
```
len1 = len(dec1)
len2 = len(dec2)
```
len1 = {}
len2 = {}
'''
            self.full = '''```
full1 = int1 + dec1
full2 = int2 + dec2
```
full1 = '{}' + '{}' = '{}'
full2 = '{}' + '{}' = '{}'
'''
            self.multiply_integer = '''```
result = multiply_integer(full1, full2)
```
enter function: multiply_integer
'''
            self.exit_function = '''exit function: multiply_integer
result = '{}'
'''
            self.find_dot = '''```
_var_res_ = find_dot(result, len1+len2)
```
enter function: find_dot
'''
            self.exit_function2 = '''exit function: find_dot
_var_res_ = '{}'
'''
            self.ret = '''```
return _var_res_
```
return '{}'
'''
        
        if dataset_name == 'multiply_hard_Fraction_Fraction_Fraction' or dataset_name == 'multiply_easy_Fraction_Fraction_Fraction':
            self.question = 'Multiply two fractions: {} and {}'
            self.rule = '''def multiply_fraction(frac1, frac2):
    num1, den1 = frac1.split('/')
    num2, den2 = frac2.split('/')
    num = multiply_integer(num1, num2)
    den = multiply_integer(den1, den2)
    gcd_num_den = gcd_integer(num, den)
    num = floordiv(num, gcd_num_den)
    den = floordiv(den, gcd_num_den)
    result = num + '/' + den
    return result
'''         
            self.initialize = '''frac1 = '{}'
frac2 = '{}'
```
num1, den1 = frac1.split('/')
num2, den2 = frac2.split('/')
```
num1 = '{}'
den1 = '{}'
num2 = '{}'
den2 = '{}'
'''
            self.calc_num = '''```
num = multiply_integer(num1, num2)
```
enter function: multiply_integer
'''
            self.exit_function1 = '''exit function: multiply_integer
num = '{}'
'''
            self.calc_den = '''```
den = multiply_integer(den1, den2)
```
enter function: multiply_integer
'''
            self.exit_function2 = '''exit function: multiply_integer
den = '{}'
'''
            self.calc_gcd = '''```
gcd_num_den = gcd_integer(num, den)
```
enter function: gcd_integer
'''
            self.exit_function3 = '''exit function: gcd_integer
gcd_num_den = '{}'
'''
            self.calc_num2 = '''```
num = floordiv(num, gcd_num_den)
```
enter function: floordiv
'''
            self.exit_function4 = '''exit function: floordiv
num = '{}'
'''
            self.calc_den2 = '''```
den = floordiv(den, gcd_num_den)
```
enter function: floordiv
'''
            self.exit_function5 = '''exit function: floordiv
den = '{}'
'''
            self.calc_result = '''```
result = num + '/' + den
```
result = '{}' + '/' + '{}' = '{}'
'''
            self.ret = '''```
return result
```
return '{}'
'''            

        if dataset_name == 'multiply_hard_ScientificNotation_ScientificNotation_ScientificNotation'or dataset_name == 'multiply_easy_ScientificNotation_ScientificNotation_ScientificNotation':
            self.question = 'Multiply two numbers: {} and {}'
            self.rule = '''
def multiply_ScientificNotation(num1, num2):
    base1, exp1 = num1.split('e')
    base2, exp2 = num2.split('e')
    result_base = multiply_float(base1, base2)
    result_exp = int(exp1) + int(exp2)
    int_part, dec_part = result_base.split('.')
    while len(int_part) > 1:
        dec_part = int_part[-1] + dec_part
        int_part = int_part[:-1]
        result_exp += 1
    dec_part = dec_part.rstrip('0') or '0'
    result_base = int_part + '.' + dec_part
    result = result_base + 'e' + str(result_exp)
    return result
'''
            self.initialize = '''num1 = '{}'
num2 = '{}'
```
base1, exp1 = num1.split('e')
base2, exp2 = num2.split('e')
```
base1 = '{}'
exp1 = '{}'
base2 = '{}'
exp2 = '{}'
'''
            self.multiply_float = '''```
result_base = multiply_float(base1, base2)
```
enter function: multiply_float
'''
            self.exit_function = '''exit function: multiply_float
result_base = '{}'
'''
            self.result_exp = '''```
result_exp = int(exp1) + int(exp2)
```
result_exp = {} + {} = {}
'''
            self.split = '''```
int_part, dec_part = result_base.split('.')
```
int_part = '{}'
dec_part = '{}'
'''
            self.enter = '''```
while len(int_part) > 1:
```
check the stop criterion
enter the loop
'''
            self.update_parts = '''```
dec_part = int_part[-1] + dec_part
int_part = int_part[:-1]
```
dec_part = '{}' + '{}' = '{}'
int_part = '{}'
'''
            self.update_exp = '''```
result_exp += 1
```
result_exp = {} + 1 = {}
'''
            self.out = '''```
while len(int_part) > 1:
```
check the stop criterion
end the loop
'''
            self.rstrip = '''```
dec_part = dec_part.rstrip('0') or '0'
```
dec_part = '{}'
'''
            self.update_result_base = '''```
result_base = int_part + '.' + dec_part
```
result_base = '{}' + '.' + '{}' = '{}'
'''
            self.merge = '''```
result = result_base + 'e' + str(result_exp)
```
result = '{}' + 'e' + '{}' = '{}'
'''
            self.ret = '''```
return result
```
return '{}'
'''   
        
        
        if dataset_name == "digit_max_Integer_Integer_Integer":
            self.question = 'Compare two numbers digit by digit and return the larger digit at each position, treating any missing digits as 0:  {} and {}'
            self.rule = '''def max(num1, num2):
    result = ''
    while num1 and num2:
        digit1 = num1[-1]
        digit2 = num2[-1]
        result = str(max(digit1, digit2)) + result
        num1 = num1[:-1]
        num2 = num2[:-1]
    result = num1 + num2 + result
    return result
'''
            self.initialize = '''num1 = '{}'
num2 = '{}'
result = ''
'''
            self.enter = '''```
while num1 and num2:
```
check the stop criterion
enter the loop
'''
            self.not_enter = '''```
while num1 and num2
```
check the stop criterion
end the loop
'''
            self.last_digit = '''```
digit1 = num1[-1]
digit2 = num2[-1]
```
num1 = '{}'
num2 = '{}'
digit1 = '{}'
digit2 = '{}'
'''
            self.max = '''```
result = str(max(digit1, digit2)) + result
```
max(digit1, digit2) = {}
result = '{}' + '{}' = '{}'
'''
            self.update_nums = '''```
num1 = num1[:-1]
num2 = num2[:-1]
```
num1 = '{}'
num2 = '{}'
'''
            self.update_rest = '''```
result = num1 + num2 + result
```
result = '{}' + '{}' + '{}' = '{}'
'''
            self.ret = '''```
return result
```
reeturn '{}'
'''
        
        
        if dataset_name == 'get_digit_Integer_int_int':
            self.question = 'Get the digit at position {} of the number {}'
            self.rule = '''
def get_digit_int(num, pos):
    cnt = 0
    while cnt < pos:
        num = num[1:]
        cnt += 1
    return num[0]
'''
            self.initialize = '''num = '{}'
pos = {}
cnt = 0
'''
            self.enter = '''```
while cnt < pos:
```
check the stop criterion
enter the loop
'''
            self.not_enter = '''```
while cnt < pos:
```
check the stop criterion
end the loop
'''
            self.update_num = '''```
num = num[1:]
```
num = '{}'
'''
            self.update_cnt = '''```
cnt += 1
```
cnt = {} + 1 = {}
'''
            self.ret = '''```
return num[0]
```
return '{}'
'''
        
        if dataset_name == "get_digit_Float_int_int":
            self.question = 'Get the digit at position {} of the number {}'
            self.rule = '''
def get_digit_float(num, pos):
    # remove dot
    cnt = 0
    num = num.replace('.', '')
    while cnt < pos:
        num = num[1:]
        cnt += 1
    return num[0]            
'''
            self.initialize = '''num = '{}'
pos = {}
cnt = 0
'''
            self.remove_dot = '''```
num = num.replace('.', '')
```
num = '{}'
'''
            self.enter = '''```
while cnt < pos:
```
check the stop criterion
enter the loop
'''
            self.not_enter = '''```
while cnt < pos:
```
check the stop criterion
end the loop
'''
            self.update_num = '''```
num = num[1:]
```
num = '{}'
'''
            self.update_cnt = '''```
cnt += 1
```
cnt = {} + 1 = {}
'''
            self.ret = '''```
return num[0]
```
return '{}'
'''
        
        if dataset_name == "length_Integer_none_int":
            self.question = 'Get the length of the number {}'
            self.rule = '''
def length(num):
    result = 0
    while num:
        result += 1
        num = num[1:]
    return result
'''
            self.initialize = '''num = '{}'
result = 0
'''
            self.enter = '''```
while num:
```
check the stop criterion
enter the loop
'''
            self.not_enter = '''```
while num:
```
check the stop criterion
end the loop
'''
            self.update_result = '''```
result += 1
```
result = {} + 1 = {}
'''
            self.update_nums = '''```
num = num[1:]
```
num = '{}'
'''
            self.ret = '''```
return result
```
return {}
'''
            
        if dataset_name == "floordiv_Integer_Integer_Integer":
            self.question = 'Divide two numbers: {} and {}'
            self.rule = '''
def floordiv(_var1_, _var2_):
    _var_res_ = ''
    num_now = 0
    while _var1_:
        num_now = num_now * 10 + int(_var1_[0])
        digit = 0
        while num_now and num_now >= _var2_:
            num_now = num_now - _var2_
            digit += 1
        _var_res_ = _var_res_ + str(digit)
        _var1_ = _var1_[1:]
    _var_res_ = _var_res_.lstrip('0') or '0'
    return _var_res_
'''
            self.initialize = '''_var1_ = '{}'
_var2_ = '{}'
_var_res_ = ''
num_now = ''
'''
            self.enter = '''```
while _var1_:
```
check the stop criterion
enter the loop
'''
            self.not_enter = '''```
while _var1_:
```
check the stop criterion
end the loop
'''
            self.update_num_now = '''```
num_now = num_now * 10 + int(_var1_[0])
```
num_now = '{}'
'''
            self.digit_init = '''```
digit = 0
```
digit = 0
'''
            self.brief = '''```
digit = num_now // num2
num_now = num_now % num2
```
digit = {} // {} = {}
num_now = {} % {} = {}
'''
            self.enter_digit = '''```
while num_now and num_now >= _var2_:
```
check the stop criterion
enter the loop
'''
            self.not_enter_digit = '''```
while num_now and num_now >= _var2_:
```
check the stop criterion
end the loop
'''
            self.update_num_now2 = '''```
num_now = num_now - _var2_
```
num_now = {} - {} = {}
'''
            self.update_digit = '''```
digit += 1
```
digit = {} + 1 = {}
'''
            self.update_result = '''```
_var_res_ = _var_res_ + str(digit)
```
_var_res_ = '{}' + '{}' = '{}'
'''
            self.update_nums = '''```
_var1_ = _var1_[1:]
```
_var1_ = '{}'
'''
            self.lstrip = '''```
_var_res_ = _var_res_.lstrip('0') or '0'
```
_var_res_ = '{}'
'''
            self.ret = '''```
return _var_res_
```
return '{}'
'''
       
        if dataset_name == 'mod_Integer_Integer_Integer' or dataset_name == 'mod_easy_Integer_Integer_Integer':
            self.question = 'Divide two numbers: {} and {}'
            self.rule = '''
def mod_integer(_var1_, _var2_):
    q = floordiv(_var1_, _var2_)
    temp = multiply_integer(q, _var2_)
    r = sub_integer(_var1_, temp)
    return r
'''
            self.initialize = '''_var1_ = '{}'
_var2_ = '{}'
'''
            self.calc_q = '''```
q = floordiv(_var1_, _var2_)
```
enter function: floordiv
'''
            self.exit_function1 = '''exit function: floordiv
q = '{}'
'''
            self.calc_temp = '''```
temp = multiply_integer(q, _var2_)
```
enter function: multiply_integer
'''
            self.exit_function2 = '''exit function: multiply_integer
temp = '{}'
'''
            self.calc_r = '''```
r = sub_integer(_var1_, temp)
```
enter function: sub_integer
'''
            self.exit_function3 = '''exit function: sub_integer
r = '{}'
'''
            self.ret = '''```
return r
```
return '{}'
'''

