## Copyright (C) 2018 Talha Hanif Butt
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} trainGradientDescent (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Talha Hanif Butt <talha@talha-Inspiron-15-7000-Gaming>
## Created: 2018-11-13

function [wts] = trainGradientDescent (X,Y,learningRate,momentum)

	wts = randn(size(X)(2),1);	% randomly initializing weights
	
	for i = 1:100
		
	  
    r = (size(X)(1)-1).*rand(size(X)(1),1) + 1;
	  
		r = round(r);
		
		X = X(r,:);
		
		Y = Y(r,:);
	
		t = X*wts;
		
		dw = momentum.*wts + (learningRate*X.'*(t-Y))/size(X)(1);
			
		wts = wts + dw;
		  
	endfor	


endfunction
