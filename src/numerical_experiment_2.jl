#--------------------------------local reduction------------------------------------
using JuMP
using Ipopt
include("local_reduction_pr.jl")

var_size = 5 # change size for 1, 5, 10, 25, 50, 100, 200, 300, 400, 500, 700, 1000
xL = [-10*ones(var_size); -Inf] #lower bound of x
xU = [10*ones(var_size); Inf]
yL = [-5*ones(var_size)]
yU = [5*ones(var_size)]

SIP_obj(x) = x[var_size + 1] #Objective function
SIP_cons_1(x,y) = (x[1:var_size] - y[1:var_size])' * (x[1:var_size] - y[1:var_size]) - x[var_size + 1]
SIP_cons = [SIP_cons_1]

y_k = [0.5*ones(var_size)]
@time local_reduction_pr(SIP_obj, SIP_cons, xL, xU, yL, yU, y_k; max_iter=50)

#-------------------------------------- EAGO.jl----------------------------------------
using JuMP
using EAGO

var_size = 5 # # change size for 1, 5, 10, 25, 50, 100, 200, 300, 400, 500, 700, 1000
xL = [-10*ones(var_size); -var_size*11^2] #lower bound of x
xU = [10*ones(var_size); var_size*11^2]
yL = -5*ones(var_size)
yU = 5*ones(var_size)
SIP_obj(x) = x[var_size + 1] #Objective function
SIP_cons_1(x, y) = ([x[1],x[2]] - [y[1],y[2]])' * ([x[1],x[2]] - [y[1],y[2]]) - x[var_size + 1]

sip_result = @time sip_solve(SIPRes(), xL, xU, yL, yU, SIP_obj, Any[SIP_cons_1], abs_tolerance = 1e-5)
println("The global minimum of the semi-infinite program is between: $(sip_result.lower_bound) and $(sip_result.upper_bound).")
println("The global minimum is attained at: x = $(sip_result.xsol).")
println("Is the problem feasible? $(sip_result.feasibility).")

#---------------------------------Constraint discretization------------------------------
using JuMP
using Ipopt
include("constraint_discretization.jl")

N = 101
var_size = 5 #maximum size within allowable time limit of 1000s: 5 

xL = [-10*ones(var_size); -Inf] #lower bound of x
xU = [10*ones(var_size); Inf]
yL = [-5*ones(var_size)]
yU = [5*ones(var_size)]
SIP_obj(x) = x[var_size + 1] #Objective function
SIP_cons_1(x,y) = (x[1:var_size] - y)' * (x[1:var_size] - y) - x[var_size + 1]
SIP_cons = [SIP_cons_1]

@time constraint_discretization(SIP_obj, SIP_cons, xL, xU, yL, yU, N)
